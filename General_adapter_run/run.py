# -*- coding: utf-8 -*-
"""
Run an LSTM model to predict chloride concentrations in Delft-FEWS.

Based on https://github.com/BasWullems/salt_intrusion_lstm/tree/main
Work by Bas Wullems

Created on Thu Dec 10 14:46:31 2024.
@author: Jurian Beunk (Deltares)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from datetime import timedelta
import random
from pathlib import Path
from fewsio import pi
import logging
import joblib

# Fix the random seed to ensure reproducible results.
np.random.seed(1)
tf.random.set_seed(2)
random.seed(3)


def get_logger(diag_xml_path: Path = Path("."), log_level=logging.INFO):
    logger = logging.getLogger("SaltiSolutions")
    if not logger.hasHandlers() and not any(
        (isinstance(h, logging.StreamHandler) for h in logger.handlers)
    ):
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if not any((isinstance(h, pi.DiagHandler) for h in logger.handlers)):
        handler = pi.DiagHandler(diag_xml_path)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def ensemble_forecast(
    models: list,
    data: pd.DataFrame,
    nfuture: int,
    saltvars: pd.Index,
    qtyvars: pd.Index,
):
    """
    Create a number of forecasts using an ensemble of models.

    Parameters
    ----------
    models : list of Functional
        Ensemble of LSTM models.
    data : DataFrame
        Input data with dates as indices and variables as columns.
    nfuture : int
        Maximum number of days to predict ahead.
    saltvars : Index
        Columns in 'data' to be used as salt input variables. Must match the
        number of salt variables used in the training of 'models'.

    # TODO: assert condition ##################################################

    qtyvars : Index
        Columns in 'data' to be used as quantity input variables. Must match
        the number of quantity variables used in the training of 'models'.

    # TODO: assert condition ##################################################

    Returns
    -------
    forecast : Array of float
        Array with shape models * lead time * issue times * salt variables.
        Contains predicted values of chloride concentrations, ginven in
        normalized values with respect to the statistics of the training
        dataset.
    forecast_real : Array of float
        Array with shape models * lead time * issue times * salt variables.
        Transformed back to real chloride concentrations, using statistics of
        the training dataset .
    """
    # set up tensor structure as model*lead time*issue time*variables
    forecast = np.empty(
        (len(models), nfuture, data.shape[0] - nfuture - N_PAST, len(saltvars))
    )
    forecast.fill(np.nan)
    forecast_real = np.empty(
        (len(models), nfuture, data.shape[0] - nfuture - N_PAST, len(saltvars))
    )
    forecast_real.fill(np.nan)

    # We have 12 features with chloride concentrations and 11 features with
    # other variables (here called quantity or qty variables). These must be
    # treated differently in the model and are therefore split into two
    # separate tensors.
    # For the salt data, the input consists of measurements of the N_PAST
    # preceding days. For the quantity data, we also include the measurement of
    # the next day, as a proxy for a forecast.

    for m in range(NUMMODELS):
        """
        TODO: when N_PAST=5, nfuture=1 and data.shape[0]=6, the dimension for salt_in etc
        will be (0, 5, 12). This means not forecast will be produced. Is this correct? 
        """
        model = models[m]
        salt_in = np.empty(
            (data.shape[0] - N_PAST - nfuture, N_PAST, len(saltvars))
        )
        salt_in.fill(np.nan)
        qty_in = np.empty(
            (data.shape[0] - N_PAST - nfuture, N_PAST + 1, len(qtyvars))
        )
        qty_in.fill(np.nan)

        for i in range(N_PAST, data.shape[0] - nfuture):
            salt_in[i - N_PAST, :, :] = np.array(
                data[saltvars][i - N_PAST : i]
            )
            qty_in[i - N_PAST, :, :] = np.array(
                data[qtyvars][i - N_PAST : i + 1]
            )

        # Create a one day ahead forecast
        forecast[m, 0, :, :] = model.predict([salt_in, qty_in])
        # Backtransform the forecast from normalized scores to real
        # concentrations.
        forecast_copies = np.hstack(
            [forecast[m, 0, :, :], forecast[m, 0, :, 1:]]
        )
        forecast_real[m, 0, :, :] = scaler.inverse_transform(forecast_copies)[
            :, 0:12
        ]

        # We create a new salt_in dataset by taking all but the first
        # observation used for that issue time. Then append the forecast we
        # just made to this observation sequence. We repeat this procedure
        # until we have reached the desired forecasting horizon given by
        # nfuture. We hereby shift the window of observations one day at the
        # time, with the last value being the result of the previous
        # forecasting step.

        for j in range(1, nfuture):
            for i in range(N_PAST, data.shape[0] - nfuture):
                salt_in[i - N_PAST, :, :] = np.vstack(
                    (
                        salt_in[i - N_PAST, 1:, :],
                        forecast[m, j - 1, i - N_PAST, :],
                    )
                )
                qty_in[i - N_PAST, :, :] = np.array(
                    data[qtyvars][i - N_PAST + j : i + j + 1]
                )
            forecast[m, j, :, :] = model.predict([salt_in, qty_in])
            forecast_copies = np.hstack(
                [forecast[m, j, :, :], forecast[m, j, :, 1:]]
            )
            forecast_real[m, j, :, :] = scaler.inverse_transform(
                forecast_copies
            )[:, 0:12]

    return forecast, forecast_real


if __name__ == "__main__":
    # Set paths
    root_dir = Path(__file__).parent
    input_dir = root_dir / "from_fews"
    output_dir = root_dir / "to_fews"
    output_file_path = output_dir / "output.xml"
    model_weights_dir = root_dir / "model_weights"
    scaler_path = root_dir / "scaler/scaler.pkl"

    # User settings
    NUMMODELS = 15  # number of models in the ensemble
    N_PAST = 5  # number of days in the past used to make a prediction
    N_FUTURE = 1  # number of days in the future for which to make a prediction
    output_parameter_id = "S.simulated.Cl"

    # Set up logger
    logger = get_logger(diag_xml_path=(output_dir / "diag.xml"))

    # Load input data
    input_pi = input_df = pi.Timeseries(
        (input_dir / "input_hydrometeo.xml"), binary=False
    )

    input_df = input_pi.to_dataframe()

    # Select relevant features from the dataframe, in correct order
    select_features = [
        #         "ClKr400Min",
        #         "ClKr400Mean",
        #         "ClKr400Max",
        ("Krimpen a/d IJssel NAP -4.0m", "S.meting.Cl", frozenset({"min"})),
        ("Krimpen a/d IJssel NAP -4.0m", "S.meting.Cl", frozenset({})),
        ("Krimpen a/d IJssel NAP -4.0m", "S.meting.Cl", frozenset({"max"})),
        #         "ClKr550Min",
        #         "ClKr550Mean",
        #         "ClKr550Max",
        ("Krimpen a/d IJssel NAP -5.5m", "S.meting.Cl", frozenset({"min"})),
        ("Krimpen a/d IJssel NAP -5.5m", "S.meting.Cl", frozenset({})),
        ("Krimpen a/d IJssel NAP -5.5m", "S.meting.Cl", frozenset({"max"})),
        #         "ClLkh250Min",
        #         "ClLkh250Mean",
        #         "ClLkh250Max",
        ("Lekhaven NAP -2.5m", "S.meting.Cl", frozenset({"min"})),
        ("Lekhaven NAP -2.5m", "S.meting.Cl", frozenset({})),
        ("Lekhaven NAP -2.5m", "S.meting.Cl", frozenset({"max"})),
        #         "ClLkh700Min",
        #         "ClLkh700Mean",
        #         "ClLkh700Max",
        ("Lekhaven NAP -7.0m", "S.meting.Cl", frozenset({"min"})),
        ("Lekhaven NAP -7.0m", "S.meting.Cl", frozenset({})),
        ("Lekhaven NAP -7.0m", "S.meting.Cl", frozenset({"max"})),
        #         "HDrdMean"
        # Is this correct?
        ("dordrecht", "H.voorspeld", frozenset()),
        #         "HHvhMean",
        ("hoekvanholland", "H.voorspeld", frozenset({})),
        #         "HKrMin",
        #         "HKrMean",
        #         "HKrMax",
        ("krimpen_ad_ijssel", "H.voorspeld", frozenset({"min"})),
        ("krimpen_ad_ijssel", "H.voorspeld", frozenset({})),
        ("krimpen_ad_ijssel", "H.voorspeld", frozenset({"max"})),
        #         "HVlaMean",
        ("VLAA", "H.voorspeld", frozenset({})),
        #         "QHagMean",
        #         "QLobMean",
        #         "QTielMean",
        ("hagestein_boven", "Q.voorspeld", frozenset({})),
        ("lobith", "Q.voorspeld", frozenset({})),
        ("tiel", "Q.voorspeld", frozenset({})),
        # TODO Thierry / Mo
        # North-Sourth and East-West components are required inputs
        # However, only direction as one vector is provided.
        # For now, duplicate
        #         "WindEW",
        #         "WindNS",
        ("Rotterdam_luchthaven", "Wind.dir.voorspeld", frozenset({})),
        ("Rotterdam_luchthaven", "Wind.dir.voorspeld", frozenset({})),
    ]

    # Select features in sorted manner in place
    input_df = input_df.loc[:, select_features]

    # Drop unnesssary levels inplace
    input_df = input_df.droplevel(["parameter_id", "qualifier_ids"], axis=1)

    # Check the shape
    if input_df.shape != (7, 23):
        msg = f"Aborting inference, shape of input {input_df.shape}. Expected (7, 23)"
        raise ValueError(msg)

    # # Check for missing values
    # if input_df.isna().any().any():
    #     msg = "Aborting inference, NaN values are not accepted in the input. Please check your data."
    #     raise ValueError(msg)

    # TODO Thierry remove after fixing NaN values from FEWS
    input_df = input_df.fillna(0)

    # Rename the features to match naming conventions during training
    # Required by Scikit learn
    input_df.columns = [
        "ClKr400Min",
        "ClKr400Mean",
        "ClKr400Max",
        "ClKr550Min",
        "ClKr550Mean",
        "ClKr550Max",
        "ClLkh250Min",
        "ClLkh250Mean",
        "ClLkh250Max",
        "ClLkh700Min",
        "ClLkh700Mean",
        "ClLkh700Max",
        "HDrdMean",
        "HHvhMean",
        "HKrMin",
        "HKrMean",
        "HKrMax",
        "HVlaMean",
        "QHagMean",
        "QLobMean",
        "QTielMean",
        "WindEW",
        "WindNS",
    ]
    # # Create features
    # features_table = pd.read_csv("Data\\Features.csv", index_col=0)
    # features_table = features_table.interpolate()
    # dates = pd.to_datetime(features_table.Time)
    # features_table.Time = pd.to_datetime(features_table["Time"])
    # features_table = features_table.set_index("Time")
    # features_table = features_table[

    # ]
    # variables = features_table.columns  # Extract variable names

    # split = features_table.index.get_loc(datetime(2018, 1, 1, 0, 0, 0))
    # train = features_table.iloc[:split, :]

    # Scale data
    scaler = joblib.load(scaler_path)
    input_df_scaled = pd.DataFrame(
        scaler.transform(input_df),
        index=input_df.index,
        columns=input_df.columns,
    )

    # Load the models from model_weights_dir
    models = [
        keras.models.load_model((model_weights_dir / f"LSTM_{x}"))
        for x in range(NUMMODELS)
    ]

    # Shape (15, 1, 1, 12)
    variables = input_df.columns
    forecast, forecast_real = ensemble_forecast(
        models,
        input_df_scaled,  # the scaled input dataframe
        N_FUTURE,
        input_df_scaled.columns[0:12],
        input_df_scaled.columns[12:23],
    )

    # Create multi-index series for output with shape.
    # Note that the t0_index has lenght 1, since operationally we will
    # make one forecast at a time.
    multi_index = pd.MultiIndex.from_product(
        [
            list(range(NUMMODELS)),  # Model dimension
            [1],  # t0 dimension
            list(range(N_FUTURE)),  # lead time dimension
            list(range(12)),  # variable dimension
        ],
        names=["model", "t0", "leadtime", "variable"],
    )

    # Create series by ravelling the array in C-style order
    # (last index changing fastest)
    series = pd.Series(data=forecast_real.ravel(), index=multi_index)

    # Unstack the variable level
    df = series.unstack(3)

    # Write to pi-xml as final step
    output = pi.Timeseries(
        str(output_file_path),
        binary=False,
        make_new_file=True,
    )

    # The output time axis is one day ahead of the last timestep
    t0 = input_df.index[-1].to_pydatetime()

    output.times = [t0]
    output.dt = input_pi.dt
    output.forecast_datetime = t0
    output.timezone = input_pi.timezone
    output.contains_ensemble = True
    output.ensemble_size = 15

    # For each variable / location (12)
    for e, tid in enumerate(select_features[:12]):
        location = tid[0]
        parameter = output_parameter_id
        qualifier = tid[2]

        # Construct a new TimeseriesId
        if "min" in qualifier:
            tid = pi.TimeseriesId(location, parameter, "min")
        elif "max" in qualifier:
            tid = pi.TimeseriesId(location, parameter, "max")
        else:
            tid = pi.TimeseriesId(location, parameter)

        data = series.loc[(slice(None), slice(None), slice(None), e)].values

        # For each model output (1 datapoint) (14 models)
        for ee, item in enumerate(data):
            print(ee)
            output.set(
                timeseries_id=tid,
                new_values=np.array([item]),
                unit="mg/l",
                ensemble_member=ee,
            )

    # Write it all to file
    output.write()

    try:
        pass
    except Exception as e:
        msg = "Salti prediction failed. Error is: {}".format(e)
        logger.error(msg, exc_info=True)
        logger.handlers[0].close()
        raise e
