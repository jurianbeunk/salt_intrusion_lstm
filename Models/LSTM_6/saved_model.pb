��!
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Č 
|
salt_pred/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namesalt_pred/kernel
u
$salt_pred/kernel/Read/ReadVariableOpReadVariableOpsalt_pred/kernel*
_output_shapes

:@*
dtype0
t
salt_pred/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namesalt_pred/bias
m
"salt_pred/bias/Read/ReadVariableOpReadVariableOpsalt_pred/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
salt_seq/lstm_cell_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*-
shared_namesalt_seq/lstm_cell_42/kernel
�
0salt_seq/lstm_cell_42/kernel/Read/ReadVariableOpReadVariableOpsalt_seq/lstm_cell_42/kernel*
_output_shapes
:	�*
dtype0
�
&salt_seq/lstm_cell_42/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*7
shared_name(&salt_seq/lstm_cell_42/recurrent_kernel
�
:salt_seq/lstm_cell_42/recurrent_kernel/Read/ReadVariableOpReadVariableOp&salt_seq/lstm_cell_42/recurrent_kernel*
_output_shapes
:	 �*
dtype0
�
salt_seq/lstm_cell_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namesalt_seq/lstm_cell_42/bias
�
.salt_seq/lstm_cell_42/bias/Read/ReadVariableOpReadVariableOpsalt_seq/lstm_cell_42/bias*
_output_shapes	
:�*
dtype0
�
qty_seq/lstm_cell_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*,
shared_nameqty_seq/lstm_cell_43/kernel
�
/qty_seq/lstm_cell_43/kernel/Read/ReadVariableOpReadVariableOpqty_seq/lstm_cell_43/kernel*
_output_shapes
:	�*
dtype0
�
%qty_seq/lstm_cell_43/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*6
shared_name'%qty_seq/lstm_cell_43/recurrent_kernel
�
9qty_seq/lstm_cell_43/recurrent_kernel/Read/ReadVariableOpReadVariableOp%qty_seq/lstm_cell_43/recurrent_kernel*
_output_shapes
:	 �*
dtype0
�
qty_seq/lstm_cell_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameqty_seq/lstm_cell_43/bias
�
-qty_seq/lstm_cell_43/bias/Read/ReadVariableOpReadVariableOpqty_seq/lstm_cell_43/bias*
_output_shapes	
:�*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
Adam/salt_pred/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/salt_pred/kernel/m
�
+Adam/salt_pred/kernel/m/Read/ReadVariableOpReadVariableOpAdam/salt_pred/kernel/m*
_output_shapes

:@*
dtype0
�
Adam/salt_pred/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/salt_pred/bias/m
{
)Adam/salt_pred/bias/m/Read/ReadVariableOpReadVariableOpAdam/salt_pred/bias/m*
_output_shapes
:*
dtype0
�
#Adam/salt_seq/lstm_cell_42/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*4
shared_name%#Adam/salt_seq/lstm_cell_42/kernel/m
�
7Adam/salt_seq/lstm_cell_42/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/salt_seq/lstm_cell_42/kernel/m*
_output_shapes
:	�*
dtype0
�
-Adam/salt_seq/lstm_cell_42/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*>
shared_name/-Adam/salt_seq/lstm_cell_42/recurrent_kernel/m
�
AAdam/salt_seq/lstm_cell_42/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp-Adam/salt_seq/lstm_cell_42/recurrent_kernel/m*
_output_shapes
:	 �*
dtype0
�
!Adam/salt_seq/lstm_cell_42/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/salt_seq/lstm_cell_42/bias/m
�
5Adam/salt_seq/lstm_cell_42/bias/m/Read/ReadVariableOpReadVariableOp!Adam/salt_seq/lstm_cell_42/bias/m*
_output_shapes	
:�*
dtype0
�
"Adam/qty_seq/lstm_cell_43/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/qty_seq/lstm_cell_43/kernel/m
�
6Adam/qty_seq/lstm_cell_43/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/qty_seq/lstm_cell_43/kernel/m*
_output_shapes
:	�*
dtype0
�
,Adam/qty_seq/lstm_cell_43/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*=
shared_name.,Adam/qty_seq/lstm_cell_43/recurrent_kernel/m
�
@Adam/qty_seq/lstm_cell_43/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/qty_seq/lstm_cell_43/recurrent_kernel/m*
_output_shapes
:	 �*
dtype0
�
 Adam/qty_seq/lstm_cell_43/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/qty_seq/lstm_cell_43/bias/m
�
4Adam/qty_seq/lstm_cell_43/bias/m/Read/ReadVariableOpReadVariableOp Adam/qty_seq/lstm_cell_43/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/salt_pred/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/salt_pred/kernel/v
�
+Adam/salt_pred/kernel/v/Read/ReadVariableOpReadVariableOpAdam/salt_pred/kernel/v*
_output_shapes

:@*
dtype0
�
Adam/salt_pred/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/salt_pred/bias/v
{
)Adam/salt_pred/bias/v/Read/ReadVariableOpReadVariableOpAdam/salt_pred/bias/v*
_output_shapes
:*
dtype0
�
#Adam/salt_seq/lstm_cell_42/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*4
shared_name%#Adam/salt_seq/lstm_cell_42/kernel/v
�
7Adam/salt_seq/lstm_cell_42/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/salt_seq/lstm_cell_42/kernel/v*
_output_shapes
:	�*
dtype0
�
-Adam/salt_seq/lstm_cell_42/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*>
shared_name/-Adam/salt_seq/lstm_cell_42/recurrent_kernel/v
�
AAdam/salt_seq/lstm_cell_42/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp-Adam/salt_seq/lstm_cell_42/recurrent_kernel/v*
_output_shapes
:	 �*
dtype0
�
!Adam/salt_seq/lstm_cell_42/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/salt_seq/lstm_cell_42/bias/v
�
5Adam/salt_seq/lstm_cell_42/bias/v/Read/ReadVariableOpReadVariableOp!Adam/salt_seq/lstm_cell_42/bias/v*
_output_shapes	
:�*
dtype0
�
"Adam/qty_seq/lstm_cell_43/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/qty_seq/lstm_cell_43/kernel/v
�
6Adam/qty_seq/lstm_cell_43/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/qty_seq/lstm_cell_43/kernel/v*
_output_shapes
:	�*
dtype0
�
,Adam/qty_seq/lstm_cell_43/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*=
shared_name.,Adam/qty_seq/lstm_cell_43/recurrent_kernel/v
�
@Adam/qty_seq/lstm_cell_43/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/qty_seq/lstm_cell_43/recurrent_kernel/v*
_output_shapes
:	 �*
dtype0
�
 Adam/qty_seq/lstm_cell_43/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/qty_seq/lstm_cell_43/bias/v
�
4Adam/qty_seq/lstm_cell_43/bias/v/Read/ReadVariableOpReadVariableOp Adam/qty_seq/lstm_cell_43/bias/v*
_output_shapes	
:�*
dtype0

NoOpNoOp
�G
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�G
value�GB�G B�G
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
�
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses*
�
cell

state_spec
	variables
trainable_variables
regularization_losses
 	keras_api
!_random_generator
"__call__
*#&call_and_return_all_conditional_losses*
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(_random_generator
)__call__
**&call_and_return_all_conditional_losses* 
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/_random_generator
0__call__
*1&call_and_return_all_conditional_losses* 
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses* 
�

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses*
�
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_rate8m�9m�Em�Fm�Gm�Hm�Im�Jm�8v�9v�Ev�Fv�Gv�Hv�Iv�Jv�*
<
E0
F1
G2
H3
I4
J5
86
97*
<
E0
F1
G2
H3
I4
J5
86
97*
* 
�
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Pserving_default* 
�
Q
state_size

Ekernel
Frecurrent_kernel
Gbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V_random_generator
W__call__
*X&call_and_return_all_conditional_losses*
* 

E0
F1
G2*

E0
F1
G2*
* 
�

Ystates
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
�
_
state_size

Hkernel
Irecurrent_kernel
Jbias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d_random_generator
e__call__
*f&call_and_return_all_conditional_losses*
* 

H0
I1
J2*

H0
I1
J2*
* 
�

gstates
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
$	variables
%trainable_variables
&regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
+	variables
,trainable_variables
-regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEsalt_pred/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEsalt_pred/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

80
91*

80
91*
* 
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEsalt_seq/lstm_cell_42/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&salt_seq/lstm_cell_42/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEsalt_seq/lstm_cell_42/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEqty_seq/lstm_cell_43/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%qty_seq/lstm_cell_43/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEqty_seq/lstm_cell_43/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
0
1
2
3
4
5
6
7*

�0*
* 
* 
* 
* 

E0
F1
G2*

E0
F1
G2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
R	variables
Strainable_variables
Tregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 

H0
I1
J2*

H0
I1
J2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

�total

�count
�	variables
�	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
�}
VARIABLE_VALUEAdam/salt_pred/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/salt_pred/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/salt_seq/lstm_cell_42/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/salt_seq/lstm_cell_42/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/salt_seq/lstm_cell_42/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/qty_seq/lstm_cell_43/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/qty_seq/lstm_cell_43/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/qty_seq/lstm_cell_43/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/salt_pred/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/salt_pred/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/salt_seq/lstm_cell_42/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/salt_seq/lstm_cell_42/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/salt_seq/lstm_cell_42/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/qty_seq/lstm_cell_43/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/qty_seq/lstm_cell_43/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/qty_seq/lstm_cell_43/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_quantity_dataPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
serving_default_salt_dataPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_quantity_dataserving_default_salt_dataqty_seq/lstm_cell_43/kernel%qty_seq/lstm_cell_43/recurrent_kernelqty_seq/lstm_cell_43/biassalt_seq/lstm_cell_42/kernel&salt_seq/lstm_cell_42/recurrent_kernelsalt_seq/lstm_cell_42/biassalt_pred/kernelsalt_pred/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_487044
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$salt_pred/kernel/Read/ReadVariableOp"salt_pred/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp0salt_seq/lstm_cell_42/kernel/Read/ReadVariableOp:salt_seq/lstm_cell_42/recurrent_kernel/Read/ReadVariableOp.salt_seq/lstm_cell_42/bias/Read/ReadVariableOp/qty_seq/lstm_cell_43/kernel/Read/ReadVariableOp9qty_seq/lstm_cell_43/recurrent_kernel/Read/ReadVariableOp-qty_seq/lstm_cell_43/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/salt_pred/kernel/m/Read/ReadVariableOp)Adam/salt_pred/bias/m/Read/ReadVariableOp7Adam/salt_seq/lstm_cell_42/kernel/m/Read/ReadVariableOpAAdam/salt_seq/lstm_cell_42/recurrent_kernel/m/Read/ReadVariableOp5Adam/salt_seq/lstm_cell_42/bias/m/Read/ReadVariableOp6Adam/qty_seq/lstm_cell_43/kernel/m/Read/ReadVariableOp@Adam/qty_seq/lstm_cell_43/recurrent_kernel/m/Read/ReadVariableOp4Adam/qty_seq/lstm_cell_43/bias/m/Read/ReadVariableOp+Adam/salt_pred/kernel/v/Read/ReadVariableOp)Adam/salt_pred/bias/v/Read/ReadVariableOp7Adam/salt_seq/lstm_cell_42/kernel/v/Read/ReadVariableOpAAdam/salt_seq/lstm_cell_42/recurrent_kernel/v/Read/ReadVariableOp5Adam/salt_seq/lstm_cell_42/bias/v/Read/ReadVariableOp6Adam/qty_seq/lstm_cell_43/kernel/v/Read/ReadVariableOp@Adam/qty_seq/lstm_cell_43/recurrent_kernel/v/Read/ReadVariableOp4Adam/qty_seq/lstm_cell_43/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_488675
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesalt_pred/kernelsalt_pred/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratesalt_seq/lstm_cell_42/kernel&salt_seq/lstm_cell_42/recurrent_kernelsalt_seq/lstm_cell_42/biasqty_seq/lstm_cell_43/kernel%qty_seq/lstm_cell_43/recurrent_kernelqty_seq/lstm_cell_43/biastotalcountAdam/salt_pred/kernel/mAdam/salt_pred/bias/m#Adam/salt_seq/lstm_cell_42/kernel/m-Adam/salt_seq/lstm_cell_42/recurrent_kernel/m!Adam/salt_seq/lstm_cell_42/bias/m"Adam/qty_seq/lstm_cell_43/kernel/m,Adam/qty_seq/lstm_cell_43/recurrent_kernel/m Adam/qty_seq/lstm_cell_43/bias/mAdam/salt_pred/kernel/vAdam/salt_pred/bias/v#Adam/salt_seq/lstm_cell_42/kernel/v-Adam/salt_seq/lstm_cell_42/recurrent_kernel/v!Adam/salt_seq/lstm_cell_42/bias/v"Adam/qty_seq/lstm_cell_43/kernel/v,Adam/qty_seq/lstm_cell_43/recurrent_kernel/v Adam/qty_seq/lstm_cell_43/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_488778��
�"
�
while_body_485034
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_42_485058_0:	�.
while_lstm_cell_42_485060_0:	 �*
while_lstm_cell_42_485062_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_42_485058:	�,
while_lstm_cell_42_485060:	 �(
while_lstm_cell_42_485062:	���*while/lstm_cell_42/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
*while/lstm_cell_42/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_42_485058_0while_lstm_cell_42_485060_0while_lstm_cell_42_485062_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_42_layer_call_and_return_conditional_losses_484975�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_42/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :����
while/Identity_4Identity3while/lstm_cell_42/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_5Identity3while/lstm_cell_42/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:��������� y

while/NoOpNoOp+^while/lstm_cell_42/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_42_485058while_lstm_cell_42_485058_0"8
while_lstm_cell_42_485060while_lstm_cell_42_485060_0"8
while_lstm_cell_42_485062while_lstm_cell_42_485062_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2X
*while/lstm_cell_42/StatefulPartitionedCall*while/lstm_cell_42/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�8
�
C__inference_qty_seq_layer_call_and_return_conditional_losses_485453

inputs&
lstm_cell_43_485371:	�&
lstm_cell_43_485373:	 �"
lstm_cell_43_485375:	�
identity��$lstm_cell_43/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
$lstm_cell_43/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_43_485371lstm_cell_43_485373lstm_cell_43_485375*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_43_layer_call_and_return_conditional_losses_485325n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_43_485371lstm_cell_43_485373lstm_cell_43_485375*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_485384*
condR
while_cond_485383*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� u
NoOpNoOp%^lstm_cell_43/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_43/StatefulPartitionedCall$lstm_cell_43/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
H__inference_lstm_cell_42_layer_call_and_return_conditional_losses_484975

inputs

states
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	 �.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:��������� V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:��������� U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:��������� N
ReluRelusplit:output:2*
T0*'
_output_shapes
:��������� _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:��������� T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:��������� V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:��������� K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:��������� c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:��������� :��������� : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates
��
�
C__inference_model_6_layer_call_and_return_conditional_losses_486713
inputs_0
inputs_1F
3qty_seq_lstm_cell_43_matmul_readvariableop_resource:	�H
5qty_seq_lstm_cell_43_matmul_1_readvariableop_resource:	 �C
4qty_seq_lstm_cell_43_biasadd_readvariableop_resource:	�G
4salt_seq_lstm_cell_42_matmul_readvariableop_resource:	�I
6salt_seq_lstm_cell_42_matmul_1_readvariableop_resource:	 �D
5salt_seq_lstm_cell_42_biasadd_readvariableop_resource:	�:
(salt_pred_matmul_readvariableop_resource:@7
)salt_pred_biasadd_readvariableop_resource:
identity��+qty_seq/lstm_cell_43/BiasAdd/ReadVariableOp�*qty_seq/lstm_cell_43/MatMul/ReadVariableOp�,qty_seq/lstm_cell_43/MatMul_1/ReadVariableOp�qty_seq/while� salt_pred/BiasAdd/ReadVariableOp�salt_pred/MatMul/ReadVariableOp�,salt_seq/lstm_cell_42/BiasAdd/ReadVariableOp�+salt_seq/lstm_cell_42/MatMul/ReadVariableOp�-salt_seq/lstm_cell_42/MatMul_1/ReadVariableOp�salt_seq/whileE
qty_seq/ShapeShapeinputs_1*
T0*
_output_shapes
:e
qty_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
qty_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
qty_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
qty_seq/strided_sliceStridedSliceqty_seq/Shape:output:0$qty_seq/strided_slice/stack:output:0&qty_seq/strided_slice/stack_1:output:0&qty_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
qty_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
qty_seq/zeros/packedPackqty_seq/strided_slice:output:0qty_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
qty_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
qty_seq/zerosFillqty_seq/zeros/packed:output:0qty_seq/zeros/Const:output:0*
T0*'
_output_shapes
:��������� Z
qty_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
qty_seq/zeros_1/packedPackqty_seq/strided_slice:output:0!qty_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
qty_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
qty_seq/zeros_1Fillqty_seq/zeros_1/packed:output:0qty_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� k
qty_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
qty_seq/transpose	Transposeinputs_1qty_seq/transpose/perm:output:0*
T0*+
_output_shapes
:���������T
qty_seq/Shape_1Shapeqty_seq/transpose:y:0*
T0*
_output_shapes
:g
qty_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
qty_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
qty_seq/strided_slice_1StridedSliceqty_seq/Shape_1:output:0&qty_seq/strided_slice_1/stack:output:0(qty_seq/strided_slice_1/stack_1:output:0(qty_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#qty_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
qty_seq/TensorArrayV2TensorListReserve,qty_seq/TensorArrayV2/element_shape:output:0 qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
=qty_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
/qty_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorqty_seq/transpose:y:0Fqty_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���g
qty_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
qty_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
qty_seq/strided_slice_2StridedSliceqty_seq/transpose:y:0&qty_seq/strided_slice_2/stack:output:0(qty_seq/strided_slice_2/stack_1:output:0(qty_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
*qty_seq/lstm_cell_43/MatMul/ReadVariableOpReadVariableOp3qty_seq_lstm_cell_43_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
qty_seq/lstm_cell_43/MatMulMatMul qty_seq/strided_slice_2:output:02qty_seq/lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,qty_seq/lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOp5qty_seq_lstm_cell_43_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
qty_seq/lstm_cell_43/MatMul_1MatMulqty_seq/zeros:output:04qty_seq/lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
qty_seq/lstm_cell_43/addAddV2%qty_seq/lstm_cell_43/MatMul:product:0'qty_seq/lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
+qty_seq/lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOp4qty_seq_lstm_cell_43_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
qty_seq/lstm_cell_43/BiasAddBiasAddqty_seq/lstm_cell_43/add:z:03qty_seq/lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������f
$qty_seq/lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
qty_seq/lstm_cell_43/splitSplit-qty_seq/lstm_cell_43/split/split_dim:output:0%qty_seq/lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split~
qty_seq/lstm_cell_43/SigmoidSigmoid#qty_seq/lstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� �
qty_seq/lstm_cell_43/Sigmoid_1Sigmoid#qty_seq/lstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� �
qty_seq/lstm_cell_43/mulMul"qty_seq/lstm_cell_43/Sigmoid_1:y:0qty_seq/zeros_1:output:0*
T0*'
_output_shapes
:��������� x
qty_seq/lstm_cell_43/ReluRelu#qty_seq/lstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
qty_seq/lstm_cell_43/mul_1Mul qty_seq/lstm_cell_43/Sigmoid:y:0'qty_seq/lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� �
qty_seq/lstm_cell_43/add_1AddV2qty_seq/lstm_cell_43/mul:z:0qty_seq/lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� �
qty_seq/lstm_cell_43/Sigmoid_2Sigmoid#qty_seq/lstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� u
qty_seq/lstm_cell_43/Relu_1Reluqty_seq/lstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
qty_seq/lstm_cell_43/mul_2Mul"qty_seq/lstm_cell_43/Sigmoid_2:y:0)qty_seq/lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� v
%qty_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
qty_seq/TensorArrayV2_1TensorListReserve.qty_seq/TensorArrayV2_1/element_shape:output:0 qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���N
qty_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 qty_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
qty_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
qty_seq/whileWhile#qty_seq/while/loop_counter:output:0)qty_seq/while/maximum_iterations:output:0qty_seq/time:output:0 qty_seq/TensorArrayV2_1:handle:0qty_seq/zeros:output:0qty_seq/zeros_1:output:0 qty_seq/strided_slice_1:output:0?qty_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:03qty_seq_lstm_cell_43_matmul_readvariableop_resource5qty_seq_lstm_cell_43_matmul_1_readvariableop_resource4qty_seq_lstm_cell_43_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
qty_seq_while_body_486480*%
condR
qty_seq_while_cond_486479*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
8qty_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
*qty_seq/TensorArrayV2Stack/TensorListStackTensorListStackqty_seq/while:output:3Aqty_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0p
qty_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
qty_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
qty_seq/strided_slice_3StridedSlice3qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0&qty_seq/strided_slice_3/stack:output:0(qty_seq/strided_slice_3/stack_1:output:0(qty_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maskm
qty_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
qty_seq/transpose_1	Transpose3qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0!qty_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� c
qty_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    F
salt_seq/ShapeShapeinputs_0*
T0*
_output_shapes
:f
salt_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
salt_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
salt_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
salt_seq/strided_sliceStridedSlicesalt_seq/Shape:output:0%salt_seq/strided_slice/stack:output:0'salt_seq/strided_slice/stack_1:output:0'salt_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
salt_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
salt_seq/zeros/packedPacksalt_seq/strided_slice:output:0 salt_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
salt_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
salt_seq/zerosFillsalt_seq/zeros/packed:output:0salt_seq/zeros/Const:output:0*
T0*'
_output_shapes
:��������� [
salt_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
salt_seq/zeros_1/packedPacksalt_seq/strided_slice:output:0"salt_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
salt_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
salt_seq/zeros_1Fill salt_seq/zeros_1/packed:output:0salt_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� l
salt_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
salt_seq/transpose	Transposeinputs_0 salt_seq/transpose/perm:output:0*
T0*+
_output_shapes
:���������V
salt_seq/Shape_1Shapesalt_seq/transpose:y:0*
T0*
_output_shapes
:h
salt_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 salt_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
salt_seq/strided_slice_1StridedSlicesalt_seq/Shape_1:output:0'salt_seq/strided_slice_1/stack:output:0)salt_seq/strided_slice_1/stack_1:output:0)salt_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$salt_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
salt_seq/TensorArrayV2TensorListReserve-salt_seq/TensorArrayV2/element_shape:output:0!salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
>salt_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0salt_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsalt_seq/transpose:y:0Gsalt_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���h
salt_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 salt_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
salt_seq/strided_slice_2StridedSlicesalt_seq/transpose:y:0'salt_seq/strided_slice_2/stack:output:0)salt_seq/strided_slice_2/stack_1:output:0)salt_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
+salt_seq/lstm_cell_42/MatMul/ReadVariableOpReadVariableOp4salt_seq_lstm_cell_42_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
salt_seq/lstm_cell_42/MatMulMatMul!salt_seq/strided_slice_2:output:03salt_seq/lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-salt_seq/lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOp6salt_seq_lstm_cell_42_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
salt_seq/lstm_cell_42/MatMul_1MatMulsalt_seq/zeros:output:05salt_seq/lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
salt_seq/lstm_cell_42/addAddV2&salt_seq/lstm_cell_42/MatMul:product:0(salt_seq/lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
,salt_seq/lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOp5salt_seq_lstm_cell_42_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
salt_seq/lstm_cell_42/BiasAddBiasAddsalt_seq/lstm_cell_42/add:z:04salt_seq/lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������g
%salt_seq/lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
salt_seq/lstm_cell_42/splitSplit.salt_seq/lstm_cell_42/split/split_dim:output:0&salt_seq/lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split�
salt_seq/lstm_cell_42/SigmoidSigmoid$salt_seq/lstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� �
salt_seq/lstm_cell_42/Sigmoid_1Sigmoid$salt_seq/lstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� �
salt_seq/lstm_cell_42/mulMul#salt_seq/lstm_cell_42/Sigmoid_1:y:0salt_seq/zeros_1:output:0*
T0*'
_output_shapes
:��������� z
salt_seq/lstm_cell_42/ReluRelu$salt_seq/lstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
salt_seq/lstm_cell_42/mul_1Mul!salt_seq/lstm_cell_42/Sigmoid:y:0(salt_seq/lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� �
salt_seq/lstm_cell_42/add_1AddV2salt_seq/lstm_cell_42/mul:z:0salt_seq/lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� �
salt_seq/lstm_cell_42/Sigmoid_2Sigmoid$salt_seq/lstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� w
salt_seq/lstm_cell_42/Relu_1Relusalt_seq/lstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
salt_seq/lstm_cell_42/mul_2Mul#salt_seq/lstm_cell_42/Sigmoid_2:y:0*salt_seq/lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� w
&salt_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
salt_seq/TensorArrayV2_1TensorListReserve/salt_seq/TensorArrayV2_1/element_shape:output:0!salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���O
salt_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!salt_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
salt_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
salt_seq/whileWhile$salt_seq/while/loop_counter:output:0*salt_seq/while/maximum_iterations:output:0salt_seq/time:output:0!salt_seq/TensorArrayV2_1:handle:0salt_seq/zeros:output:0salt_seq/zeros_1:output:0!salt_seq/strided_slice_1:output:0@salt_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:04salt_seq_lstm_cell_42_matmul_readvariableop_resource6salt_seq_lstm_cell_42_matmul_1_readvariableop_resource5salt_seq_lstm_cell_42_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
salt_seq_while_body_486619*&
condR
salt_seq_while_cond_486618*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
9salt_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
+salt_seq/TensorArrayV2Stack/TensorListStackTensorListStacksalt_seq/while:output:3Bsalt_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0q
salt_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 salt_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
salt_seq/strided_slice_3StridedSlice4salt_seq/TensorArrayV2Stack/TensorListStack:tensor:0'salt_seq/strided_slice_3/stack:output:0)salt_seq/strided_slice_3/stack_1:output:0)salt_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maskn
salt_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
salt_seq/transpose_1	Transpose4salt_seq/TensorArrayV2Stack/TensorListStack:tensor:0"salt_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� d
salt_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    t
salt_seq_2/IdentityIdentity!salt_seq/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� r
qty_seq_2/IdentityIdentity qty_seq/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� U
pattern/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
pattern/concatConcatV2salt_seq_2/Identity:output:0qty_seq_2/Identity:output:0pattern/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������@�
salt_pred/MatMul/ReadVariableOpReadVariableOp(salt_pred_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
salt_pred/MatMulMatMulpattern/concat:output:0'salt_pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 salt_pred/BiasAdd/ReadVariableOpReadVariableOp)salt_pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
salt_pred/BiasAddBiasAddsalt_pred/MatMul:product:0(salt_pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
IdentityIdentitysalt_pred/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^qty_seq/lstm_cell_43/BiasAdd/ReadVariableOp+^qty_seq/lstm_cell_43/MatMul/ReadVariableOp-^qty_seq/lstm_cell_43/MatMul_1/ReadVariableOp^qty_seq/while!^salt_pred/BiasAdd/ReadVariableOp ^salt_pred/MatMul/ReadVariableOp-^salt_seq/lstm_cell_42/BiasAdd/ReadVariableOp,^salt_seq/lstm_cell_42/MatMul/ReadVariableOp.^salt_seq/lstm_cell_42/MatMul_1/ReadVariableOp^salt_seq/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2Z
+qty_seq/lstm_cell_43/BiasAdd/ReadVariableOp+qty_seq/lstm_cell_43/BiasAdd/ReadVariableOp2X
*qty_seq/lstm_cell_43/MatMul/ReadVariableOp*qty_seq/lstm_cell_43/MatMul/ReadVariableOp2\
,qty_seq/lstm_cell_43/MatMul_1/ReadVariableOp,qty_seq/lstm_cell_43/MatMul_1/ReadVariableOp2
qty_seq/whileqty_seq/while2D
 salt_pred/BiasAdd/ReadVariableOp salt_pred/BiasAdd/ReadVariableOp2B
salt_pred/MatMul/ReadVariableOpsalt_pred/MatMul/ReadVariableOp2\
,salt_seq/lstm_cell_42/BiasAdd/ReadVariableOp,salt_seq/lstm_cell_42/BiasAdd/ReadVariableOp2Z
+salt_seq/lstm_cell_42/MatMul/ReadVariableOp+salt_seq/lstm_cell_42/MatMul/ReadVariableOp2^
-salt_seq/lstm_cell_42/MatMul_1/ReadVariableOp-salt_seq/lstm_cell_42/MatMul_1/ReadVariableOp2 
salt_seq/whilesalt_seq/while:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs/1
�J
�
D__inference_salt_seq_layer_call_and_return_conditional_losses_485763

inputs>
+lstm_cell_42_matmul_readvariableop_resource:	�@
-lstm_cell_42_matmul_1_readvariableop_resource:	 �;
,lstm_cell_42_biasadd_readvariableop_resource:	�
identity��#lstm_cell_42/BiasAdd/ReadVariableOp�"lstm_cell_42/MatMul/ReadVariableOp�$lstm_cell_42/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_42/MatMul/ReadVariableOpReadVariableOp+lstm_cell_42_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_42/MatMulMatMulstrided_slice_2:output:0*lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_42_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
lstm_cell_42/MatMul_1MatMulzeros:output:0,lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_42/addAddV2lstm_cell_42/MatMul:product:0lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_42_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_42/BiasAddBiasAddlstm_cell_42/add:z:0+lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_42/splitSplit%lstm_cell_42/split/split_dim:output:0lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitn
lstm_cell_42/SigmoidSigmoidlstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� p
lstm_cell_42/Sigmoid_1Sigmoidlstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� w
lstm_cell_42/mulMullstm_cell_42/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� h
lstm_cell_42/ReluRelulstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
lstm_cell_42/mul_1Mullstm_cell_42/Sigmoid:y:0lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� {
lstm_cell_42/add_1AddV2lstm_cell_42/mul:z:0lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� p
lstm_cell_42/Sigmoid_2Sigmoidlstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� e
lstm_cell_42/Relu_1Relulstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
lstm_cell_42/mul_2Mullstm_cell_42/Sigmoid_2:y:0!lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_42_matmul_readvariableop_resource-lstm_cell_42_matmul_1_readvariableop_resource,lstm_cell_42_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_485679*
condR
while_cond_485678*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp$^lstm_cell_42/BiasAdd/ReadVariableOp#^lstm_cell_42/MatMul/ReadVariableOp%^lstm_cell_42/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_42/BiasAdd/ReadVariableOp#lstm_cell_42/BiasAdd/ReadVariableOp2H
"lstm_cell_42/MatMul/ReadVariableOp"lstm_cell_42/MatMul/ReadVariableOp2L
$lstm_cell_42/MatMul_1/ReadVariableOp$lstm_cell_42/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_485033
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_485033___redundant_placeholder04
0while_while_cond_485033___redundant_placeholder14
0while_while_cond_485033___redundant_placeholder24
0while_while_cond_485033___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
c
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_485783

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�8
�
while_body_487147
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_42_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_42_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_42_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_42_matmul_readvariableop_resource:	�F
3while_lstm_cell_42_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_42_biasadd_readvariableop_resource:	���)while/lstm_cell_42/BiasAdd/ReadVariableOp�(while/lstm_cell_42/MatMul/ReadVariableOp�*while/lstm_cell_42/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_42/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_42/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_42_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
while/lstm_cell_42/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_42/addAddV2#while/lstm_cell_42/MatMul:product:0%while/lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_42_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_42/BiasAddBiasAddwhile/lstm_cell_42/add:z:01while/lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_42/splitSplit+while/lstm_cell_42/split/split_dim:output:0#while/lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitz
while/lstm_cell_42/SigmoidSigmoid!while/lstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_42/Sigmoid_1Sigmoid!while/lstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mulMul while/lstm_cell_42/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� t
while/lstm_cell_42/ReluRelu!while/lstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mul_1Mulwhile/lstm_cell_42/Sigmoid:y:0%while/lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/add_1AddV2while/lstm_cell_42/mul:z:0while/lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_42/Sigmoid_2Sigmoid!while/lstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� q
while/lstm_cell_42/Relu_1Reluwhile/lstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mul_2Mul while/lstm_cell_42/Sigmoid_2:y:0'while/lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_42/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���y
while/Identity_4Identitywhile/lstm_cell_42/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� y
while/Identity_5Identitywhile/lstm_cell_42/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp*^while/lstm_cell_42/BiasAdd/ReadVariableOp)^while/lstm_cell_42/MatMul/ReadVariableOp+^while/lstm_cell_42/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_42_biasadd_readvariableop_resource4while_lstm_cell_42_biasadd_readvariableop_resource_0"l
3while_lstm_cell_42_matmul_1_readvariableop_resource5while_lstm_cell_42_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_42_matmul_readvariableop_resource3while_lstm_cell_42_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_42/BiasAdd/ReadVariableOp)while/lstm_cell_42/BiasAdd/ReadVariableOp2T
(while/lstm_cell_42/MatMul/ReadVariableOp(while/lstm_cell_42/MatMul/ReadVariableOp2X
*while/lstm_cell_42/MatMul_1/ReadVariableOp*while/lstm_cell_42/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�	
�
E__inference_salt_pred_layer_call_and_return_conditional_losses_485804

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�J
�
D__inference_salt_seq_layer_call_and_return_conditional_losses_487231
inputs_0>
+lstm_cell_42_matmul_readvariableop_resource:	�@
-lstm_cell_42_matmul_1_readvariableop_resource:	 �;
,lstm_cell_42_biasadd_readvariableop_resource:	�
identity��#lstm_cell_42/BiasAdd/ReadVariableOp�"lstm_cell_42/MatMul/ReadVariableOp�$lstm_cell_42/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_42/MatMul/ReadVariableOpReadVariableOp+lstm_cell_42_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_42/MatMulMatMulstrided_slice_2:output:0*lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_42_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
lstm_cell_42/MatMul_1MatMulzeros:output:0,lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_42/addAddV2lstm_cell_42/MatMul:product:0lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_42_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_42/BiasAddBiasAddlstm_cell_42/add:z:0+lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_42/splitSplit%lstm_cell_42/split/split_dim:output:0lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitn
lstm_cell_42/SigmoidSigmoidlstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� p
lstm_cell_42/Sigmoid_1Sigmoidlstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� w
lstm_cell_42/mulMullstm_cell_42/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� h
lstm_cell_42/ReluRelulstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
lstm_cell_42/mul_1Mullstm_cell_42/Sigmoid:y:0lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� {
lstm_cell_42/add_1AddV2lstm_cell_42/mul:z:0lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� p
lstm_cell_42/Sigmoid_2Sigmoidlstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� e
lstm_cell_42/Relu_1Relulstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
lstm_cell_42/mul_2Mullstm_cell_42/Sigmoid_2:y:0!lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_42_matmul_readvariableop_resource-lstm_cell_42_matmul_1_readvariableop_resource,lstm_cell_42_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_487147*
condR
while_cond_487146*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp$^lstm_cell_42/BiasAdd/ReadVariableOp#^lstm_cell_42/MatMul/ReadVariableOp%^lstm_cell_42/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_42/BiasAdd/ReadVariableOp#lstm_cell_42/BiasAdd/ReadVariableOp2H
"lstm_cell_42/MatMul/ReadVariableOp"lstm_cell_42/MatMul/ReadVariableOp2L
$lstm_cell_42/MatMul_1/ReadVariableOp$lstm_cell_42/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
F
*__inference_qty_seq_2_layer_call_fn_488308

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_485783`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
qty_seq_while_cond_486479,
(qty_seq_while_qty_seq_while_loop_counter2
.qty_seq_while_qty_seq_while_maximum_iterations
qty_seq_while_placeholder
qty_seq_while_placeholder_1
qty_seq_while_placeholder_2
qty_seq_while_placeholder_3.
*qty_seq_while_less_qty_seq_strided_slice_1D
@qty_seq_while_qty_seq_while_cond_486479___redundant_placeholder0D
@qty_seq_while_qty_seq_while_cond_486479___redundant_placeholder1D
@qty_seq_while_qty_seq_while_cond_486479___redundant_placeholder2D
@qty_seq_while_qty_seq_while_cond_486479___redundant_placeholder3
qty_seq_while_identity
�
qty_seq/while/LessLessqty_seq_while_placeholder*qty_seq_while_less_qty_seq_strided_slice_1*
T0*
_output_shapes
: [
qty_seq/while/IdentityIdentityqty_seq/while/Less:z:0*
T0
*
_output_shapes
: "9
qty_seq_while_identityqty_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
while_cond_485528
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_485528___redundant_placeholder04
0while_while_cond_485528___redundant_placeholder14
0while_while_cond_485528___redundant_placeholder24
0while_while_cond_485528___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
while_cond_487289
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_487289___redundant_placeholder04
0while_while_cond_487289___redundant_placeholder14
0while_while_cond_487289___redundant_placeholder24
0while_while_cond_487289___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
H__inference_lstm_cell_42_layer_call_and_return_conditional_losses_484829

inputs

states
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	 �.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:��������� V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:��������� U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:��������� N
ReluRelusplit:output:2*
T0*'
_output_shapes
:��������� _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:��������� T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:��������� V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:��������� K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:��������� c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:��������� :��������� : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates
�J
�
D__inference_salt_seq_layer_call_and_return_conditional_losses_486049

inputs>
+lstm_cell_42_matmul_readvariableop_resource:	�@
-lstm_cell_42_matmul_1_readvariableop_resource:	 �;
,lstm_cell_42_biasadd_readvariableop_resource:	�
identity��#lstm_cell_42/BiasAdd/ReadVariableOp�"lstm_cell_42/MatMul/ReadVariableOp�$lstm_cell_42/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_42/MatMul/ReadVariableOpReadVariableOp+lstm_cell_42_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_42/MatMulMatMulstrided_slice_2:output:0*lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_42_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
lstm_cell_42/MatMul_1MatMulzeros:output:0,lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_42/addAddV2lstm_cell_42/MatMul:product:0lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_42_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_42/BiasAddBiasAddlstm_cell_42/add:z:0+lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_42/splitSplit%lstm_cell_42/split/split_dim:output:0lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitn
lstm_cell_42/SigmoidSigmoidlstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� p
lstm_cell_42/Sigmoid_1Sigmoidlstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� w
lstm_cell_42/mulMullstm_cell_42/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� h
lstm_cell_42/ReluRelulstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
lstm_cell_42/mul_1Mullstm_cell_42/Sigmoid:y:0lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� {
lstm_cell_42/add_1AddV2lstm_cell_42/mul:z:0lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� p
lstm_cell_42/Sigmoid_2Sigmoidlstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� e
lstm_cell_42/Relu_1Relulstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
lstm_cell_42/mul_2Mullstm_cell_42/Sigmoid_2:y:0!lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_42_matmul_readvariableop_resource-lstm_cell_42_matmul_1_readvariableop_resource,lstm_cell_42_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_485965*
condR
while_cond_485964*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp$^lstm_cell_42/BiasAdd/ReadVariableOp#^lstm_cell_42/MatMul/ReadVariableOp%^lstm_cell_42/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_42/BiasAdd/ReadVariableOp#lstm_cell_42/BiasAdd/ReadVariableOp2H
"lstm_cell_42/MatMul/ReadVariableOp"lstm_cell_42/MatMul/ReadVariableOp2L
$lstm_cell_42/MatMul_1/ReadVariableOp$lstm_cell_42/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
(__inference_model_6_layer_call_fn_486398
inputs_0
inputs_1
unknown:	�
	unknown_0:	 �
	unknown_1:	�
	unknown_2:	�
	unknown_3:	 �
	unknown_4:	�
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_485811o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs/1
�J
�
C__inference_qty_seq_layer_call_and_return_conditional_losses_488276

inputs>
+lstm_cell_43_matmul_readvariableop_resource:	�@
-lstm_cell_43_matmul_1_readvariableop_resource:	 �;
,lstm_cell_43_biasadd_readvariableop_resource:	�
identity��#lstm_cell_43/BiasAdd/ReadVariableOp�"lstm_cell_43/MatMul/ReadVariableOp�$lstm_cell_43/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_43/MatMul/ReadVariableOpReadVariableOp+lstm_cell_43_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_43/MatMulMatMulstrided_slice_2:output:0*lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_43_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
lstm_cell_43/MatMul_1MatMulzeros:output:0,lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_43/addAddV2lstm_cell_43/MatMul:product:0lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_43_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_43/BiasAddBiasAddlstm_cell_43/add:z:0+lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_43/splitSplit%lstm_cell_43/split/split_dim:output:0lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitn
lstm_cell_43/SigmoidSigmoidlstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� p
lstm_cell_43/Sigmoid_1Sigmoidlstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� w
lstm_cell_43/mulMullstm_cell_43/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� h
lstm_cell_43/ReluRelulstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
lstm_cell_43/mul_1Mullstm_cell_43/Sigmoid:y:0lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� {
lstm_cell_43/add_1AddV2lstm_cell_43/mul:z:0lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� p
lstm_cell_43/Sigmoid_2Sigmoidlstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� e
lstm_cell_43/Relu_1Relulstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
lstm_cell_43/mul_2Mullstm_cell_43/Sigmoid_2:y:0!lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_43_matmul_readvariableop_resource-lstm_cell_43_matmul_1_readvariableop_resource,lstm_cell_43_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_488192*
condR
while_cond_488191*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp$^lstm_cell_43/BiasAdd/ReadVariableOp#^lstm_cell_43/MatMul/ReadVariableOp%^lstm_cell_43/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_43/BiasAdd/ReadVariableOp#lstm_cell_43/BiasAdd/ReadVariableOp2H
"lstm_cell_43/MatMul/ReadVariableOp"lstm_cell_43/MatMul/ReadVariableOp2L
$lstm_cell_43/MatMul_1/ReadVariableOp$lstm_cell_43/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�8
�
while_body_485529
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_43_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_43_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_43_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_43_matmul_readvariableop_resource:	�F
3while_lstm_cell_43_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_43_biasadd_readvariableop_resource:	���)while/lstm_cell_43/BiasAdd/ReadVariableOp�(while/lstm_cell_43/MatMul/ReadVariableOp�*while/lstm_cell_43/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_43/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_43_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_43/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_43_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
while/lstm_cell_43/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_43/addAddV2#while/lstm_cell_43/MatMul:product:0%while/lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_43_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_43/BiasAddBiasAddwhile/lstm_cell_43/add:z:01while/lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_43/splitSplit+while/lstm_cell_43/split/split_dim:output:0#while/lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitz
while/lstm_cell_43/SigmoidSigmoid!while/lstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_43/Sigmoid_1Sigmoid!while/lstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mulMul while/lstm_cell_43/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� t
while/lstm_cell_43/ReluRelu!while/lstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mul_1Mulwhile/lstm_cell_43/Sigmoid:y:0%while/lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/add_1AddV2while/lstm_cell_43/mul:z:0while/lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_43/Sigmoid_2Sigmoid!while/lstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� q
while/lstm_cell_43/Relu_1Reluwhile/lstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mul_2Mul while/lstm_cell_43/Sigmoid_2:y:0'while/lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_43/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���y
while/Identity_4Identitywhile/lstm_cell_43/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� y
while/Identity_5Identitywhile/lstm_cell_43/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp*^while/lstm_cell_43/BiasAdd/ReadVariableOp)^while/lstm_cell_43/MatMul/ReadVariableOp+^while/lstm_cell_43/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_43_biasadd_readvariableop_resource4while_lstm_cell_43_biasadd_readvariableop_resource_0"l
3while_lstm_cell_43_matmul_1_readvariableop_resource5while_lstm_cell_43_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_43_matmul_readvariableop_resource3while_lstm_cell_43_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_43/BiasAdd/ReadVariableOp)while/lstm_cell_43/BiasAdd/ReadVariableOp2T
(while/lstm_cell_43/MatMul/ReadVariableOp(while/lstm_cell_43/MatMul/ReadVariableOp2X
*while/lstm_cell_43/MatMul_1/ReadVariableOp*while/lstm_cell_43/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_lstm_cell_42_layer_call_and_return_conditional_losses_488460

inputs
states_0
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	 �.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:��������� V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:��������� U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:��������� N
ReluRelusplit:output:2*
T0*'
_output_shapes
:��������� _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:��������� T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:��������� V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:��������� K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:��������� c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:��������� :��������� : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/1
�
�
C__inference_model_6_layer_call_and_return_conditional_losses_486370
	salt_data
quantity_data!
qty_seq_486347:	�!
qty_seq_486349:	 �
qty_seq_486351:	�"
salt_seq_486354:	�"
salt_seq_486356:	 �
salt_seq_486358:	�"
salt_pred_486364:@
salt_pred_486366:
identity��qty_seq/StatefulPartitionedCall�!qty_seq_2/StatefulPartitionedCall�!salt_pred/StatefulPartitionedCall� salt_seq/StatefulPartitionedCall�"salt_seq_2/StatefulPartitionedCall�
qty_seq/StatefulPartitionedCallStatefulPartitionedCallquantity_dataqty_seq_486347qty_seq_486349qty_seq_486351*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_486214�
 salt_seq/StatefulPartitionedCallStatefulPartitionedCall	salt_datasalt_seq_486354salt_seq_486356salt_seq_486358*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_486049�
"salt_seq_2/StatefulPartitionedCallStatefulPartitionedCall)salt_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_485890�
!qty_seq_2/StatefulPartitionedCallStatefulPartitionedCall(qty_seq/StatefulPartitionedCall:output:0#^salt_seq_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_485867�
pattern/PartitionedCallPartitionedCall+salt_seq_2/StatefulPartitionedCall:output:0*qty_seq_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_pattern_layer_call_and_return_conditional_losses_485792�
!salt_pred/StatefulPartitionedCallStatefulPartitionedCall pattern/PartitionedCall:output:0salt_pred_486364salt_pred_486366*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_salt_pred_layer_call_and_return_conditional_losses_485804y
IdentityIdentity*salt_pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^qty_seq/StatefulPartitionedCall"^qty_seq_2/StatefulPartitionedCall"^salt_pred/StatefulPartitionedCall!^salt_seq/StatefulPartitionedCall#^salt_seq_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2B
qty_seq/StatefulPartitionedCallqty_seq/StatefulPartitionedCall2F
!qty_seq_2/StatefulPartitionedCall!qty_seq_2/StatefulPartitionedCall2F
!salt_pred/StatefulPartitionedCall!salt_pred/StatefulPartitionedCall2D
 salt_seq/StatefulPartitionedCall salt_seq/StatefulPartitionedCall2H
"salt_seq_2/StatefulPartitionedCall"salt_seq_2/StatefulPartitionedCall:V R
+
_output_shapes
:���������
#
_user_specified_name	salt_data:ZV
+
_output_shapes
:���������
'
_user_specified_namequantity_data
�

�
(__inference_model_6_layer_call_fn_486316
	salt_data
quantity_data
unknown:	�
	unknown_0:	 �
	unknown_1:	�
	unknown_2:	�
	unknown_3:	 �
	unknown_4:	�
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	salt_dataquantity_dataunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_486275o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:���������
#
_user_specified_name	salt_data:ZV
+
_output_shapes
:���������
'
_user_specified_namequantity_data
�
�
*__inference_salt_pred_layer_call_fn_488352

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_salt_pred_layer_call_and_return_conditional_losses_485804o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�J
�
D__inference_salt_seq_layer_call_and_return_conditional_losses_487660

inputs>
+lstm_cell_42_matmul_readvariableop_resource:	�@
-lstm_cell_42_matmul_1_readvariableop_resource:	 �;
,lstm_cell_42_biasadd_readvariableop_resource:	�
identity��#lstm_cell_42/BiasAdd/ReadVariableOp�"lstm_cell_42/MatMul/ReadVariableOp�$lstm_cell_42/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_42/MatMul/ReadVariableOpReadVariableOp+lstm_cell_42_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_42/MatMulMatMulstrided_slice_2:output:0*lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_42_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
lstm_cell_42/MatMul_1MatMulzeros:output:0,lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_42/addAddV2lstm_cell_42/MatMul:product:0lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_42_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_42/BiasAddBiasAddlstm_cell_42/add:z:0+lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_42/splitSplit%lstm_cell_42/split/split_dim:output:0lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitn
lstm_cell_42/SigmoidSigmoidlstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� p
lstm_cell_42/Sigmoid_1Sigmoidlstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� w
lstm_cell_42/mulMullstm_cell_42/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� h
lstm_cell_42/ReluRelulstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
lstm_cell_42/mul_1Mullstm_cell_42/Sigmoid:y:0lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� {
lstm_cell_42/add_1AddV2lstm_cell_42/mul:z:0lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� p
lstm_cell_42/Sigmoid_2Sigmoidlstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� e
lstm_cell_42/Relu_1Relulstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
lstm_cell_42/mul_2Mullstm_cell_42/Sigmoid_2:y:0!lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_42_matmul_readvariableop_resource-lstm_cell_42_matmul_1_readvariableop_resource,lstm_cell_42_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_487576*
condR
while_cond_487575*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp$^lstm_cell_42/BiasAdd/ReadVariableOp#^lstm_cell_42/MatMul/ReadVariableOp%^lstm_cell_42/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_42/BiasAdd/ReadVariableOp#lstm_cell_42/BiasAdd/ReadVariableOp2H
"lstm_cell_42/MatMul/ReadVariableOp"lstm_cell_42/MatMul/ReadVariableOp2L
$lstm_cell_42/MatMul_1/ReadVariableOp$lstm_cell_42/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_487575
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_487575___redundant_placeholder04
0while_while_cond_487575___redundant_placeholder14
0while_while_cond_487575___redundant_placeholder24
0while_while_cond_487575___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�J
�
C__inference_qty_seq_layer_call_and_return_conditional_losses_487847
inputs_0>
+lstm_cell_43_matmul_readvariableop_resource:	�@
-lstm_cell_43_matmul_1_readvariableop_resource:	 �;
,lstm_cell_43_biasadd_readvariableop_resource:	�
identity��#lstm_cell_43/BiasAdd/ReadVariableOp�"lstm_cell_43/MatMul/ReadVariableOp�$lstm_cell_43/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_43/MatMul/ReadVariableOpReadVariableOp+lstm_cell_43_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_43/MatMulMatMulstrided_slice_2:output:0*lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_43_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
lstm_cell_43/MatMul_1MatMulzeros:output:0,lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_43/addAddV2lstm_cell_43/MatMul:product:0lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_43_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_43/BiasAddBiasAddlstm_cell_43/add:z:0+lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_43/splitSplit%lstm_cell_43/split/split_dim:output:0lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitn
lstm_cell_43/SigmoidSigmoidlstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� p
lstm_cell_43/Sigmoid_1Sigmoidlstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� w
lstm_cell_43/mulMullstm_cell_43/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� h
lstm_cell_43/ReluRelulstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
lstm_cell_43/mul_1Mullstm_cell_43/Sigmoid:y:0lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� {
lstm_cell_43/add_1AddV2lstm_cell_43/mul:z:0lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� p
lstm_cell_43/Sigmoid_2Sigmoidlstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� e
lstm_cell_43/Relu_1Relulstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
lstm_cell_43/mul_2Mullstm_cell_43/Sigmoid_2:y:0!lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_43_matmul_readvariableop_resource-lstm_cell_43_matmul_1_readvariableop_resource,lstm_cell_43_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_487763*
condR
while_cond_487762*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp$^lstm_cell_43/BiasAdd/ReadVariableOp#^lstm_cell_43/MatMul/ReadVariableOp%^lstm_cell_43/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_43/BiasAdd/ReadVariableOp#lstm_cell_43/BiasAdd/ReadVariableOp2H
"lstm_cell_43/MatMul/ReadVariableOp"lstm_cell_43/MatMul/ReadVariableOp2L
$lstm_cell_43/MatMul_1/ReadVariableOp$lstm_cell_43/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
(__inference_qty_seq_layer_call_fn_487682
inputs_0
unknown:	�
	unknown_0:	 �
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_485453o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
-__inference_lstm_cell_43_layer_call_fn_488494

inputs
states_0
states_1
unknown:	�
	unknown_0:	 �
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_43_layer_call_and_return_conditional_losses_485325o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:��������� q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:��������� :��������� : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/1
�8
�
while_body_488192
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_43_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_43_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_43_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_43_matmul_readvariableop_resource:	�F
3while_lstm_cell_43_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_43_biasadd_readvariableop_resource:	���)while/lstm_cell_43/BiasAdd/ReadVariableOp�(while/lstm_cell_43/MatMul/ReadVariableOp�*while/lstm_cell_43/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_43/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_43_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_43/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_43_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
while/lstm_cell_43/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_43/addAddV2#while/lstm_cell_43/MatMul:product:0%while/lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_43_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_43/BiasAddBiasAddwhile/lstm_cell_43/add:z:01while/lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_43/splitSplit+while/lstm_cell_43/split/split_dim:output:0#while/lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitz
while/lstm_cell_43/SigmoidSigmoid!while/lstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_43/Sigmoid_1Sigmoid!while/lstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mulMul while/lstm_cell_43/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� t
while/lstm_cell_43/ReluRelu!while/lstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mul_1Mulwhile/lstm_cell_43/Sigmoid:y:0%while/lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/add_1AddV2while/lstm_cell_43/mul:z:0while/lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_43/Sigmoid_2Sigmoid!while/lstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� q
while/lstm_cell_43/Relu_1Reluwhile/lstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mul_2Mul while/lstm_cell_43/Sigmoid_2:y:0'while/lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_43/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���y
while/Identity_4Identitywhile/lstm_cell_43/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� y
while/Identity_5Identitywhile/lstm_cell_43/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp*^while/lstm_cell_43/BiasAdd/ReadVariableOp)^while/lstm_cell_43/MatMul/ReadVariableOp+^while/lstm_cell_43/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_43_biasadd_readvariableop_resource4while_lstm_cell_43_biasadd_readvariableop_resource_0"l
3while_lstm_cell_43_matmul_1_readvariableop_resource5while_lstm_cell_43_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_43_matmul_readvariableop_resource3while_lstm_cell_43_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_43/BiasAdd/ReadVariableOp)while/lstm_cell_43/BiasAdd/ReadVariableOp2T
(while/lstm_cell_43/MatMul/ReadVariableOp(while/lstm_cell_43/MatMul/ReadVariableOp2X
*while/lstm_cell_43/MatMul_1/ReadVariableOp*while/lstm_cell_43/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
)__inference_salt_seq_layer_call_fn_487088

inputs
unknown:	�
	unknown_0:	 �
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_486049o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_model_6_layer_call_and_return_conditional_losses_486275

inputs
inputs_1!
qty_seq_486252:	�!
qty_seq_486254:	 �
qty_seq_486256:	�"
salt_seq_486259:	�"
salt_seq_486261:	 �
salt_seq_486263:	�"
salt_pred_486269:@
salt_pred_486271:
identity��qty_seq/StatefulPartitionedCall�!qty_seq_2/StatefulPartitionedCall�!salt_pred/StatefulPartitionedCall� salt_seq/StatefulPartitionedCall�"salt_seq_2/StatefulPartitionedCall�
qty_seq/StatefulPartitionedCallStatefulPartitionedCallinputs_1qty_seq_486252qty_seq_486254qty_seq_486256*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_486214�
 salt_seq/StatefulPartitionedCallStatefulPartitionedCallinputssalt_seq_486259salt_seq_486261salt_seq_486263*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_486049�
"salt_seq_2/StatefulPartitionedCallStatefulPartitionedCall)salt_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_485890�
!qty_seq_2/StatefulPartitionedCallStatefulPartitionedCall(qty_seq/StatefulPartitionedCall:output:0#^salt_seq_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_485867�
pattern/PartitionedCallPartitionedCall+salt_seq_2/StatefulPartitionedCall:output:0*qty_seq_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_pattern_layer_call_and_return_conditional_losses_485792�
!salt_pred/StatefulPartitionedCallStatefulPartitionedCall pattern/PartitionedCall:output:0salt_pred_486269salt_pred_486271*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_salt_pred_layer_call_and_return_conditional_losses_485804y
IdentityIdentity*salt_pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^qty_seq/StatefulPartitionedCall"^qty_seq_2/StatefulPartitionedCall"^salt_pred/StatefulPartitionedCall!^salt_seq/StatefulPartitionedCall#^salt_seq_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2B
qty_seq/StatefulPartitionedCallqty_seq/StatefulPartitionedCall2F
!qty_seq_2/StatefulPartitionedCall!qty_seq_2/StatefulPartitionedCall2F
!salt_pred/StatefulPartitionedCall!salt_pred/StatefulPartitionedCall2D
 salt_seq/StatefulPartitionedCall salt_seq/StatefulPartitionedCall2H
"salt_seq_2/StatefulPartitionedCall"salt_seq_2/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
d
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_485867

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:��������� Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
-__inference_lstm_cell_43_layer_call_fn_488477

inputs
states_0
states_1
unknown:	�
	unknown_0:	 �
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_43_layer_call_and_return_conditional_losses_485179o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:��������� q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:��������� :��������� : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/1
�8
�
while_body_488049
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_43_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_43_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_43_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_43_matmul_readvariableop_resource:	�F
3while_lstm_cell_43_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_43_biasadd_readvariableop_resource:	���)while/lstm_cell_43/BiasAdd/ReadVariableOp�(while/lstm_cell_43/MatMul/ReadVariableOp�*while/lstm_cell_43/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_43/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_43_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_43/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_43_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
while/lstm_cell_43/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_43/addAddV2#while/lstm_cell_43/MatMul:product:0%while/lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_43_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_43/BiasAddBiasAddwhile/lstm_cell_43/add:z:01while/lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_43/splitSplit+while/lstm_cell_43/split/split_dim:output:0#while/lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitz
while/lstm_cell_43/SigmoidSigmoid!while/lstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_43/Sigmoid_1Sigmoid!while/lstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mulMul while/lstm_cell_43/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� t
while/lstm_cell_43/ReluRelu!while/lstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mul_1Mulwhile/lstm_cell_43/Sigmoid:y:0%while/lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/add_1AddV2while/lstm_cell_43/mul:z:0while/lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_43/Sigmoid_2Sigmoid!while/lstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� q
while/lstm_cell_43/Relu_1Reluwhile/lstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mul_2Mul while/lstm_cell_43/Sigmoid_2:y:0'while/lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_43/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���y
while/Identity_4Identitywhile/lstm_cell_43/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� y
while/Identity_5Identitywhile/lstm_cell_43/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp*^while/lstm_cell_43/BiasAdd/ReadVariableOp)^while/lstm_cell_43/MatMul/ReadVariableOp+^while/lstm_cell_43/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_43_biasadd_readvariableop_resource4while_lstm_cell_43_biasadd_readvariableop_resource_0"l
3while_lstm_cell_43_matmul_1_readvariableop_resource5while_lstm_cell_43_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_43_matmul_readvariableop_resource3while_lstm_cell_43_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_43/BiasAdd/ReadVariableOp)while/lstm_cell_43/BiasAdd/ReadVariableOp2T
(while/lstm_cell_43/MatMul/ReadVariableOp(while/lstm_cell_43/MatMul/ReadVariableOp2X
*while/lstm_cell_43/MatMul_1/ReadVariableOp*while/lstm_cell_43/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�J
�
D__inference_salt_seq_layer_call_and_return_conditional_losses_487517

inputs>
+lstm_cell_42_matmul_readvariableop_resource:	�@
-lstm_cell_42_matmul_1_readvariableop_resource:	 �;
,lstm_cell_42_biasadd_readvariableop_resource:	�
identity��#lstm_cell_42/BiasAdd/ReadVariableOp�"lstm_cell_42/MatMul/ReadVariableOp�$lstm_cell_42/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_42/MatMul/ReadVariableOpReadVariableOp+lstm_cell_42_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_42/MatMulMatMulstrided_slice_2:output:0*lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_42_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
lstm_cell_42/MatMul_1MatMulzeros:output:0,lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_42/addAddV2lstm_cell_42/MatMul:product:0lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_42_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_42/BiasAddBiasAddlstm_cell_42/add:z:0+lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_42/splitSplit%lstm_cell_42/split/split_dim:output:0lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitn
lstm_cell_42/SigmoidSigmoidlstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� p
lstm_cell_42/Sigmoid_1Sigmoidlstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� w
lstm_cell_42/mulMullstm_cell_42/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� h
lstm_cell_42/ReluRelulstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
lstm_cell_42/mul_1Mullstm_cell_42/Sigmoid:y:0lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� {
lstm_cell_42/add_1AddV2lstm_cell_42/mul:z:0lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� p
lstm_cell_42/Sigmoid_2Sigmoidlstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� e
lstm_cell_42/Relu_1Relulstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
lstm_cell_42/mul_2Mullstm_cell_42/Sigmoid_2:y:0!lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_42_matmul_readvariableop_resource-lstm_cell_42_matmul_1_readvariableop_resource,lstm_cell_42_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_487433*
condR
while_cond_487432*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp$^lstm_cell_42/BiasAdd/ReadVariableOp#^lstm_cell_42/MatMul/ReadVariableOp%^lstm_cell_42/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_42/BiasAdd/ReadVariableOp#lstm_cell_42/BiasAdd/ReadVariableOp2H
"lstm_cell_42/MatMul/ReadVariableOp"lstm_cell_42/MatMul/ReadVariableOp2L
$lstm_cell_42/MatMul_1/ReadVariableOp$lstm_cell_42/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
C__inference_model_6_layer_call_and_return_conditional_losses_487020
inputs_0
inputs_1F
3qty_seq_lstm_cell_43_matmul_readvariableop_resource:	�H
5qty_seq_lstm_cell_43_matmul_1_readvariableop_resource:	 �C
4qty_seq_lstm_cell_43_biasadd_readvariableop_resource:	�G
4salt_seq_lstm_cell_42_matmul_readvariableop_resource:	�I
6salt_seq_lstm_cell_42_matmul_1_readvariableop_resource:	 �D
5salt_seq_lstm_cell_42_biasadd_readvariableop_resource:	�:
(salt_pred_matmul_readvariableop_resource:@7
)salt_pred_biasadd_readvariableop_resource:
identity��+qty_seq/lstm_cell_43/BiasAdd/ReadVariableOp�*qty_seq/lstm_cell_43/MatMul/ReadVariableOp�,qty_seq/lstm_cell_43/MatMul_1/ReadVariableOp�qty_seq/while� salt_pred/BiasAdd/ReadVariableOp�salt_pred/MatMul/ReadVariableOp�,salt_seq/lstm_cell_42/BiasAdd/ReadVariableOp�+salt_seq/lstm_cell_42/MatMul/ReadVariableOp�-salt_seq/lstm_cell_42/MatMul_1/ReadVariableOp�salt_seq/whileE
qty_seq/ShapeShapeinputs_1*
T0*
_output_shapes
:e
qty_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
qty_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
qty_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
qty_seq/strided_sliceStridedSliceqty_seq/Shape:output:0$qty_seq/strided_slice/stack:output:0&qty_seq/strided_slice/stack_1:output:0&qty_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
qty_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
qty_seq/zeros/packedPackqty_seq/strided_slice:output:0qty_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
qty_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
qty_seq/zerosFillqty_seq/zeros/packed:output:0qty_seq/zeros/Const:output:0*
T0*'
_output_shapes
:��������� Z
qty_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
qty_seq/zeros_1/packedPackqty_seq/strided_slice:output:0!qty_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
qty_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
qty_seq/zeros_1Fillqty_seq/zeros_1/packed:output:0qty_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� k
qty_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
qty_seq/transpose	Transposeinputs_1qty_seq/transpose/perm:output:0*
T0*+
_output_shapes
:���������T
qty_seq/Shape_1Shapeqty_seq/transpose:y:0*
T0*
_output_shapes
:g
qty_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
qty_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
qty_seq/strided_slice_1StridedSliceqty_seq/Shape_1:output:0&qty_seq/strided_slice_1/stack:output:0(qty_seq/strided_slice_1/stack_1:output:0(qty_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#qty_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
qty_seq/TensorArrayV2TensorListReserve,qty_seq/TensorArrayV2/element_shape:output:0 qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
=qty_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
/qty_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorqty_seq/transpose:y:0Fqty_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���g
qty_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
qty_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
qty_seq/strided_slice_2StridedSliceqty_seq/transpose:y:0&qty_seq/strided_slice_2/stack:output:0(qty_seq/strided_slice_2/stack_1:output:0(qty_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
*qty_seq/lstm_cell_43/MatMul/ReadVariableOpReadVariableOp3qty_seq_lstm_cell_43_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
qty_seq/lstm_cell_43/MatMulMatMul qty_seq/strided_slice_2:output:02qty_seq/lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,qty_seq/lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOp5qty_seq_lstm_cell_43_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
qty_seq/lstm_cell_43/MatMul_1MatMulqty_seq/zeros:output:04qty_seq/lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
qty_seq/lstm_cell_43/addAddV2%qty_seq/lstm_cell_43/MatMul:product:0'qty_seq/lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
+qty_seq/lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOp4qty_seq_lstm_cell_43_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
qty_seq/lstm_cell_43/BiasAddBiasAddqty_seq/lstm_cell_43/add:z:03qty_seq/lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������f
$qty_seq/lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
qty_seq/lstm_cell_43/splitSplit-qty_seq/lstm_cell_43/split/split_dim:output:0%qty_seq/lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split~
qty_seq/lstm_cell_43/SigmoidSigmoid#qty_seq/lstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� �
qty_seq/lstm_cell_43/Sigmoid_1Sigmoid#qty_seq/lstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� �
qty_seq/lstm_cell_43/mulMul"qty_seq/lstm_cell_43/Sigmoid_1:y:0qty_seq/zeros_1:output:0*
T0*'
_output_shapes
:��������� x
qty_seq/lstm_cell_43/ReluRelu#qty_seq/lstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
qty_seq/lstm_cell_43/mul_1Mul qty_seq/lstm_cell_43/Sigmoid:y:0'qty_seq/lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� �
qty_seq/lstm_cell_43/add_1AddV2qty_seq/lstm_cell_43/mul:z:0qty_seq/lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� �
qty_seq/lstm_cell_43/Sigmoid_2Sigmoid#qty_seq/lstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� u
qty_seq/lstm_cell_43/Relu_1Reluqty_seq/lstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
qty_seq/lstm_cell_43/mul_2Mul"qty_seq/lstm_cell_43/Sigmoid_2:y:0)qty_seq/lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� v
%qty_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
qty_seq/TensorArrayV2_1TensorListReserve.qty_seq/TensorArrayV2_1/element_shape:output:0 qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���N
qty_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 qty_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
qty_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
qty_seq/whileWhile#qty_seq/while/loop_counter:output:0)qty_seq/while/maximum_iterations:output:0qty_seq/time:output:0 qty_seq/TensorArrayV2_1:handle:0qty_seq/zeros:output:0qty_seq/zeros_1:output:0 qty_seq/strided_slice_1:output:0?qty_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:03qty_seq_lstm_cell_43_matmul_readvariableop_resource5qty_seq_lstm_cell_43_matmul_1_readvariableop_resource4qty_seq_lstm_cell_43_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
qty_seq_while_body_486773*%
condR
qty_seq_while_cond_486772*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
8qty_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
*qty_seq/TensorArrayV2Stack/TensorListStackTensorListStackqty_seq/while:output:3Aqty_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0p
qty_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
qty_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
qty_seq/strided_slice_3StridedSlice3qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0&qty_seq/strided_slice_3/stack:output:0(qty_seq/strided_slice_3/stack_1:output:0(qty_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maskm
qty_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
qty_seq/transpose_1	Transpose3qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0!qty_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� c
qty_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    F
salt_seq/ShapeShapeinputs_0*
T0*
_output_shapes
:f
salt_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
salt_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
salt_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
salt_seq/strided_sliceStridedSlicesalt_seq/Shape:output:0%salt_seq/strided_slice/stack:output:0'salt_seq/strided_slice/stack_1:output:0'salt_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
salt_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
salt_seq/zeros/packedPacksalt_seq/strided_slice:output:0 salt_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
salt_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
salt_seq/zerosFillsalt_seq/zeros/packed:output:0salt_seq/zeros/Const:output:0*
T0*'
_output_shapes
:��������� [
salt_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
salt_seq/zeros_1/packedPacksalt_seq/strided_slice:output:0"salt_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
salt_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
salt_seq/zeros_1Fill salt_seq/zeros_1/packed:output:0salt_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� l
salt_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
salt_seq/transpose	Transposeinputs_0 salt_seq/transpose/perm:output:0*
T0*+
_output_shapes
:���������V
salt_seq/Shape_1Shapesalt_seq/transpose:y:0*
T0*
_output_shapes
:h
salt_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 salt_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
salt_seq/strided_slice_1StridedSlicesalt_seq/Shape_1:output:0'salt_seq/strided_slice_1/stack:output:0)salt_seq/strided_slice_1/stack_1:output:0)salt_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$salt_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
salt_seq/TensorArrayV2TensorListReserve-salt_seq/TensorArrayV2/element_shape:output:0!salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
>salt_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0salt_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsalt_seq/transpose:y:0Gsalt_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���h
salt_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 salt_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
salt_seq/strided_slice_2StridedSlicesalt_seq/transpose:y:0'salt_seq/strided_slice_2/stack:output:0)salt_seq/strided_slice_2/stack_1:output:0)salt_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
+salt_seq/lstm_cell_42/MatMul/ReadVariableOpReadVariableOp4salt_seq_lstm_cell_42_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
salt_seq/lstm_cell_42/MatMulMatMul!salt_seq/strided_slice_2:output:03salt_seq/lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-salt_seq/lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOp6salt_seq_lstm_cell_42_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
salt_seq/lstm_cell_42/MatMul_1MatMulsalt_seq/zeros:output:05salt_seq/lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
salt_seq/lstm_cell_42/addAddV2&salt_seq/lstm_cell_42/MatMul:product:0(salt_seq/lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
,salt_seq/lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOp5salt_seq_lstm_cell_42_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
salt_seq/lstm_cell_42/BiasAddBiasAddsalt_seq/lstm_cell_42/add:z:04salt_seq/lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������g
%salt_seq/lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
salt_seq/lstm_cell_42/splitSplit.salt_seq/lstm_cell_42/split/split_dim:output:0&salt_seq/lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split�
salt_seq/lstm_cell_42/SigmoidSigmoid$salt_seq/lstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� �
salt_seq/lstm_cell_42/Sigmoid_1Sigmoid$salt_seq/lstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� �
salt_seq/lstm_cell_42/mulMul#salt_seq/lstm_cell_42/Sigmoid_1:y:0salt_seq/zeros_1:output:0*
T0*'
_output_shapes
:��������� z
salt_seq/lstm_cell_42/ReluRelu$salt_seq/lstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
salt_seq/lstm_cell_42/mul_1Mul!salt_seq/lstm_cell_42/Sigmoid:y:0(salt_seq/lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� �
salt_seq/lstm_cell_42/add_1AddV2salt_seq/lstm_cell_42/mul:z:0salt_seq/lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� �
salt_seq/lstm_cell_42/Sigmoid_2Sigmoid$salt_seq/lstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� w
salt_seq/lstm_cell_42/Relu_1Relusalt_seq/lstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
salt_seq/lstm_cell_42/mul_2Mul#salt_seq/lstm_cell_42/Sigmoid_2:y:0*salt_seq/lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� w
&salt_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
salt_seq/TensorArrayV2_1TensorListReserve/salt_seq/TensorArrayV2_1/element_shape:output:0!salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���O
salt_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!salt_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
salt_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
salt_seq/whileWhile$salt_seq/while/loop_counter:output:0*salt_seq/while/maximum_iterations:output:0salt_seq/time:output:0!salt_seq/TensorArrayV2_1:handle:0salt_seq/zeros:output:0salt_seq/zeros_1:output:0!salt_seq/strided_slice_1:output:0@salt_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:04salt_seq_lstm_cell_42_matmul_readvariableop_resource6salt_seq_lstm_cell_42_matmul_1_readvariableop_resource5salt_seq_lstm_cell_42_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
salt_seq_while_body_486912*&
condR
salt_seq_while_cond_486911*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
9salt_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
+salt_seq/TensorArrayV2Stack/TensorListStackTensorListStacksalt_seq/while:output:3Bsalt_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0q
salt_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 salt_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
salt_seq/strided_slice_3StridedSlice4salt_seq/TensorArrayV2Stack/TensorListStack:tensor:0'salt_seq/strided_slice_3/stack:output:0)salt_seq/strided_slice_3/stack_1:output:0)salt_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maskn
salt_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
salt_seq/transpose_1	Transpose4salt_seq/TensorArrayV2Stack/TensorListStack:tensor:0"salt_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� d
salt_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
salt_seq_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
salt_seq_2/dropout/MulMul!salt_seq/strided_slice_3:output:0!salt_seq_2/dropout/Const:output:0*
T0*'
_output_shapes
:��������� i
salt_seq_2/dropout/ShapeShape!salt_seq/strided_slice_3:output:0*
T0*
_output_shapes
:�
/salt_seq_2/dropout/random_uniform/RandomUniformRandomUniform!salt_seq_2/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0*

seedf
!salt_seq_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
salt_seq_2/dropout/GreaterEqualGreaterEqual8salt_seq_2/dropout/random_uniform/RandomUniform:output:0*salt_seq_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� �
salt_seq_2/dropout/CastCast#salt_seq_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� �
salt_seq_2/dropout/Mul_1Mulsalt_seq_2/dropout/Mul:z:0salt_seq_2/dropout/Cast:y:0*
T0*'
_output_shapes
:��������� \
qty_seq_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
qty_seq_2/dropout/MulMul qty_seq/strided_slice_3:output:0 qty_seq_2/dropout/Const:output:0*
T0*'
_output_shapes
:��������� g
qty_seq_2/dropout/ShapeShape qty_seq/strided_slice_3:output:0*
T0*
_output_shapes
:�
.qty_seq_2/dropout/random_uniform/RandomUniformRandomUniform qty_seq_2/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0*

seed*
seed2e
 qty_seq_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
qty_seq_2/dropout/GreaterEqualGreaterEqual7qty_seq_2/dropout/random_uniform/RandomUniform:output:0)qty_seq_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� �
qty_seq_2/dropout/CastCast"qty_seq_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� �
qty_seq_2/dropout/Mul_1Mulqty_seq_2/dropout/Mul:z:0qty_seq_2/dropout/Cast:y:0*
T0*'
_output_shapes
:��������� U
pattern/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
pattern/concatConcatV2salt_seq_2/dropout/Mul_1:z:0qty_seq_2/dropout/Mul_1:z:0pattern/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������@�
salt_pred/MatMul/ReadVariableOpReadVariableOp(salt_pred_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
salt_pred/MatMulMatMulpattern/concat:output:0'salt_pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 salt_pred/BiasAdd/ReadVariableOpReadVariableOp)salt_pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
salt_pred/BiasAddBiasAddsalt_pred/MatMul:product:0(salt_pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
IdentityIdentitysalt_pred/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^qty_seq/lstm_cell_43/BiasAdd/ReadVariableOp+^qty_seq/lstm_cell_43/MatMul/ReadVariableOp-^qty_seq/lstm_cell_43/MatMul_1/ReadVariableOp^qty_seq/while!^salt_pred/BiasAdd/ReadVariableOp ^salt_pred/MatMul/ReadVariableOp-^salt_seq/lstm_cell_42/BiasAdd/ReadVariableOp,^salt_seq/lstm_cell_42/MatMul/ReadVariableOp.^salt_seq/lstm_cell_42/MatMul_1/ReadVariableOp^salt_seq/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2Z
+qty_seq/lstm_cell_43/BiasAdd/ReadVariableOp+qty_seq/lstm_cell_43/BiasAdd/ReadVariableOp2X
*qty_seq/lstm_cell_43/MatMul/ReadVariableOp*qty_seq/lstm_cell_43/MatMul/ReadVariableOp2\
,qty_seq/lstm_cell_43/MatMul_1/ReadVariableOp,qty_seq/lstm_cell_43/MatMul_1/ReadVariableOp2
qty_seq/whileqty_seq/while2D
 salt_pred/BiasAdd/ReadVariableOp salt_pred/BiasAdd/ReadVariableOp2B
salt_pred/MatMul/ReadVariableOpsalt_pred/MatMul/ReadVariableOp2\
,salt_seq/lstm_cell_42/BiasAdd/ReadVariableOp,salt_seq/lstm_cell_42/BiasAdd/ReadVariableOp2Z
+salt_seq/lstm_cell_42/MatMul/ReadVariableOp+salt_seq/lstm_cell_42/MatMul/ReadVariableOp2^
-salt_seq/lstm_cell_42/MatMul_1/ReadVariableOp-salt_seq/lstm_cell_42/MatMul_1/ReadVariableOp2 
salt_seq/whilesalt_seq/while:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
H__inference_lstm_cell_43_layer_call_and_return_conditional_losses_485325

inputs

states
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	 �.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:��������� V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:��������� U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:��������� N
ReluRelusplit:output:2*
T0*'
_output_shapes
:��������� _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:��������� T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:��������� V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:��������� K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:��������� c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:��������� :��������� : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates
�
�
)__inference_salt_seq_layer_call_fn_487055
inputs_0
unknown:	�
	unknown_0:	 �
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_484912o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
H__inference_lstm_cell_43_layer_call_and_return_conditional_losses_488526

inputs
states_0
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	 �.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:��������� V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:��������� U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:��������� N
ReluRelusplit:output:2*
T0*'
_output_shapes
:��������� _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:��������� T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:��������� V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:��������� K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:��������� c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:��������� :��������� : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/1
�
�
C__inference_model_6_layer_call_and_return_conditional_losses_486343
	salt_data
quantity_data!
qty_seq_486320:	�!
qty_seq_486322:	 �
qty_seq_486324:	�"
salt_seq_486327:	�"
salt_seq_486329:	 �
salt_seq_486331:	�"
salt_pred_486337:@
salt_pred_486339:
identity��qty_seq/StatefulPartitionedCall�!salt_pred/StatefulPartitionedCall� salt_seq/StatefulPartitionedCall�
qty_seq/StatefulPartitionedCallStatefulPartitionedCallquantity_dataqty_seq_486320qty_seq_486322qty_seq_486324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_485613�
 salt_seq/StatefulPartitionedCallStatefulPartitionedCall	salt_datasalt_seq_486327salt_seq_486329salt_seq_486331*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_485763�
salt_seq_2/PartitionedCallPartitionedCall)salt_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_485776�
qty_seq_2/PartitionedCallPartitionedCall(qty_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_485783�
pattern/PartitionedCallPartitionedCall#salt_seq_2/PartitionedCall:output:0"qty_seq_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_pattern_layer_call_and_return_conditional_losses_485792�
!salt_pred/StatefulPartitionedCallStatefulPartitionedCall pattern/PartitionedCall:output:0salt_pred_486337salt_pred_486339*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_salt_pred_layer_call_and_return_conditional_losses_485804y
IdentityIdentity*salt_pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^qty_seq/StatefulPartitionedCall"^salt_pred/StatefulPartitionedCall!^salt_seq/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2B
qty_seq/StatefulPartitionedCallqty_seq/StatefulPartitionedCall2F
!salt_pred/StatefulPartitionedCall!salt_pred/StatefulPartitionedCall2D
 salt_seq/StatefulPartitionedCall salt_seq/StatefulPartitionedCall:V R
+
_output_shapes
:���������
#
_user_specified_name	salt_data:ZV
+
_output_shapes
:���������
'
_user_specified_namequantity_data
�8
�
D__inference_salt_seq_layer_call_and_return_conditional_losses_484912

inputs&
lstm_cell_42_484830:	�&
lstm_cell_42_484832:	 �"
lstm_cell_42_484834:	�
identity��$lstm_cell_42/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
$lstm_cell_42/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_42_484830lstm_cell_42_484832lstm_cell_42_484834*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_42_layer_call_and_return_conditional_losses_484829n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_42_484830lstm_cell_42_484832lstm_cell_42_484834*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_484843*
condR
while_cond_484842*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� u
NoOpNoOp%^lstm_cell_42/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_42/StatefulPartitionedCall$lstm_cell_42/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
while_cond_487432
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_487432___redundant_placeholder04
0while_while_cond_487432___redundant_placeholder14
0while_while_cond_487432___redundant_placeholder24
0while_while_cond_487432___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
while_cond_487905
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_487905___redundant_placeholder04
0while_while_cond_487905___redundant_placeholder14
0while_while_cond_487905___redundant_placeholder24
0while_while_cond_487905___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�B
�

salt_seq_while_body_486912.
*salt_seq_while_salt_seq_while_loop_counter4
0salt_seq_while_salt_seq_while_maximum_iterations
salt_seq_while_placeholder 
salt_seq_while_placeholder_1 
salt_seq_while_placeholder_2 
salt_seq_while_placeholder_3-
)salt_seq_while_salt_seq_strided_slice_1_0i
esalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0O
<salt_seq_while_lstm_cell_42_matmul_readvariableop_resource_0:	�Q
>salt_seq_while_lstm_cell_42_matmul_1_readvariableop_resource_0:	 �L
=salt_seq_while_lstm_cell_42_biasadd_readvariableop_resource_0:	�
salt_seq_while_identity
salt_seq_while_identity_1
salt_seq_while_identity_2
salt_seq_while_identity_3
salt_seq_while_identity_4
salt_seq_while_identity_5+
'salt_seq_while_salt_seq_strided_slice_1g
csalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensorM
:salt_seq_while_lstm_cell_42_matmul_readvariableop_resource:	�O
<salt_seq_while_lstm_cell_42_matmul_1_readvariableop_resource:	 �J
;salt_seq_while_lstm_cell_42_biasadd_readvariableop_resource:	���2salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOp�1salt_seq/while/lstm_cell_42/MatMul/ReadVariableOp�3salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOp�
@salt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2salt_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemesalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0salt_seq_while_placeholderIsalt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
1salt_seq/while/lstm_cell_42/MatMul/ReadVariableOpReadVariableOp<salt_seq_while_lstm_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
"salt_seq/while/lstm_cell_42/MatMulMatMul9salt_seq/while/TensorArrayV2Read/TensorListGetItem:item:09salt_seq/while/lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
3salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOp>salt_seq_while_lstm_cell_42_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
$salt_seq/while/lstm_cell_42/MatMul_1MatMulsalt_seq_while_placeholder_2;salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
salt_seq/while/lstm_cell_42/addAddV2,salt_seq/while/lstm_cell_42/MatMul:product:0.salt_seq/while/lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
2salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOp=salt_seq_while_lstm_cell_42_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
#salt_seq/while/lstm_cell_42/BiasAddBiasAdd#salt_seq/while/lstm_cell_42/add:z:0:salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
+salt_seq/while/lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
!salt_seq/while/lstm_cell_42/splitSplit4salt_seq/while/lstm_cell_42/split/split_dim:output:0,salt_seq/while/lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split�
#salt_seq/while/lstm_cell_42/SigmoidSigmoid*salt_seq/while/lstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� �
%salt_seq/while/lstm_cell_42/Sigmoid_1Sigmoid*salt_seq/while/lstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� �
salt_seq/while/lstm_cell_42/mulMul)salt_seq/while/lstm_cell_42/Sigmoid_1:y:0salt_seq_while_placeholder_3*
T0*'
_output_shapes
:��������� �
 salt_seq/while/lstm_cell_42/ReluRelu*salt_seq/while/lstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
!salt_seq/while/lstm_cell_42/mul_1Mul'salt_seq/while/lstm_cell_42/Sigmoid:y:0.salt_seq/while/lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� �
!salt_seq/while/lstm_cell_42/add_1AddV2#salt_seq/while/lstm_cell_42/mul:z:0%salt_seq/while/lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� �
%salt_seq/while/lstm_cell_42/Sigmoid_2Sigmoid*salt_seq/while/lstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� �
"salt_seq/while/lstm_cell_42/Relu_1Relu%salt_seq/while/lstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
!salt_seq/while/lstm_cell_42/mul_2Mul)salt_seq/while/lstm_cell_42/Sigmoid_2:y:00salt_seq/while/lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
3salt_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemsalt_seq_while_placeholder_1salt_seq_while_placeholder%salt_seq/while/lstm_cell_42/mul_2:z:0*
_output_shapes
: *
element_dtype0:���V
salt_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
salt_seq/while/addAddV2salt_seq_while_placeholdersalt_seq/while/add/y:output:0*
T0*
_output_shapes
: X
salt_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
salt_seq/while/add_1AddV2*salt_seq_while_salt_seq_while_loop_countersalt_seq/while/add_1/y:output:0*
T0*
_output_shapes
: t
salt_seq/while/IdentityIdentitysalt_seq/while/add_1:z:0^salt_seq/while/NoOp*
T0*
_output_shapes
: �
salt_seq/while/Identity_1Identity0salt_seq_while_salt_seq_while_maximum_iterations^salt_seq/while/NoOp*
T0*
_output_shapes
: t
salt_seq/while/Identity_2Identitysalt_seq/while/add:z:0^salt_seq/while/NoOp*
T0*
_output_shapes
: �
salt_seq/while/Identity_3IdentityCsalt_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^salt_seq/while/NoOp*
T0*
_output_shapes
: :����
salt_seq/while/Identity_4Identity%salt_seq/while/lstm_cell_42/mul_2:z:0^salt_seq/while/NoOp*
T0*'
_output_shapes
:��������� �
salt_seq/while/Identity_5Identity%salt_seq/while/lstm_cell_42/add_1:z:0^salt_seq/while/NoOp*
T0*'
_output_shapes
:��������� �
salt_seq/while/NoOpNoOp3^salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOp2^salt_seq/while/lstm_cell_42/MatMul/ReadVariableOp4^salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
salt_seq_while_identity salt_seq/while/Identity:output:0"?
salt_seq_while_identity_1"salt_seq/while/Identity_1:output:0"?
salt_seq_while_identity_2"salt_seq/while/Identity_2:output:0"?
salt_seq_while_identity_3"salt_seq/while/Identity_3:output:0"?
salt_seq_while_identity_4"salt_seq/while/Identity_4:output:0"?
salt_seq_while_identity_5"salt_seq/while/Identity_5:output:0"|
;salt_seq_while_lstm_cell_42_biasadd_readvariableop_resource=salt_seq_while_lstm_cell_42_biasadd_readvariableop_resource_0"~
<salt_seq_while_lstm_cell_42_matmul_1_readvariableop_resource>salt_seq_while_lstm_cell_42_matmul_1_readvariableop_resource_0"z
:salt_seq_while_lstm_cell_42_matmul_readvariableop_resource<salt_seq_while_lstm_cell_42_matmul_readvariableop_resource_0"T
'salt_seq_while_salt_seq_strided_slice_1)salt_seq_while_salt_seq_strided_slice_1_0"�
csalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensoresalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2h
2salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOp2salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOp2f
1salt_seq/while/lstm_cell_42/MatMul/ReadVariableOp1salt_seq/while/lstm_cell_42/MatMul/ReadVariableOp2j
3salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOp3salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�J
�
D__inference_salt_seq_layer_call_and_return_conditional_losses_487374
inputs_0>
+lstm_cell_42_matmul_readvariableop_resource:	�@
-lstm_cell_42_matmul_1_readvariableop_resource:	 �;
,lstm_cell_42_biasadd_readvariableop_resource:	�
identity��#lstm_cell_42/BiasAdd/ReadVariableOp�"lstm_cell_42/MatMul/ReadVariableOp�$lstm_cell_42/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_42/MatMul/ReadVariableOpReadVariableOp+lstm_cell_42_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_42/MatMulMatMulstrided_slice_2:output:0*lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_42_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
lstm_cell_42/MatMul_1MatMulzeros:output:0,lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_42/addAddV2lstm_cell_42/MatMul:product:0lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_42_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_42/BiasAddBiasAddlstm_cell_42/add:z:0+lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_42/splitSplit%lstm_cell_42/split/split_dim:output:0lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitn
lstm_cell_42/SigmoidSigmoidlstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� p
lstm_cell_42/Sigmoid_1Sigmoidlstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� w
lstm_cell_42/mulMullstm_cell_42/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� h
lstm_cell_42/ReluRelulstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
lstm_cell_42/mul_1Mullstm_cell_42/Sigmoid:y:0lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� {
lstm_cell_42/add_1AddV2lstm_cell_42/mul:z:0lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� p
lstm_cell_42/Sigmoid_2Sigmoidlstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� e
lstm_cell_42/Relu_1Relulstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
lstm_cell_42/mul_2Mullstm_cell_42/Sigmoid_2:y:0!lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_42_matmul_readvariableop_resource-lstm_cell_42_matmul_1_readvariableop_resource,lstm_cell_42_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_487290*
condR
while_cond_487289*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp$^lstm_cell_42/BiasAdd/ReadVariableOp#^lstm_cell_42/MatMul/ReadVariableOp%^lstm_cell_42/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_42/BiasAdd/ReadVariableOp#lstm_cell_42/BiasAdd/ReadVariableOp2H
"lstm_cell_42/MatMul/ReadVariableOp"lstm_cell_42/MatMul/ReadVariableOp2L
$lstm_cell_42/MatMul_1/ReadVariableOp$lstm_cell_42/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�

�
(__inference_model_6_layer_call_fn_485830
	salt_data
quantity_data
unknown:	�
	unknown_0:	 �
	unknown_1:	�
	unknown_2:	�
	unknown_3:	 �
	unknown_4:	�
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	salt_dataquantity_dataunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_485811o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:���������
#
_user_specified_name	salt_data:ZV
+
_output_shapes
:���������
'
_user_specified_namequantity_data
�
�
while_cond_485964
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_485964___redundant_placeholder04
0while_while_cond_485964___redundant_placeholder14
0while_while_cond_485964___redundant_placeholder24
0while_while_cond_485964___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�L
�
"model_6_salt_seq_while_body_484668>
:model_6_salt_seq_while_model_6_salt_seq_while_loop_counterD
@model_6_salt_seq_while_model_6_salt_seq_while_maximum_iterations&
"model_6_salt_seq_while_placeholder(
$model_6_salt_seq_while_placeholder_1(
$model_6_salt_seq_while_placeholder_2(
$model_6_salt_seq_while_placeholder_3=
9model_6_salt_seq_while_model_6_salt_seq_strided_slice_1_0y
umodel_6_salt_seq_while_tensorarrayv2read_tensorlistgetitem_model_6_salt_seq_tensorarrayunstack_tensorlistfromtensor_0W
Dmodel_6_salt_seq_while_lstm_cell_42_matmul_readvariableop_resource_0:	�Y
Fmodel_6_salt_seq_while_lstm_cell_42_matmul_1_readvariableop_resource_0:	 �T
Emodel_6_salt_seq_while_lstm_cell_42_biasadd_readvariableop_resource_0:	�#
model_6_salt_seq_while_identity%
!model_6_salt_seq_while_identity_1%
!model_6_salt_seq_while_identity_2%
!model_6_salt_seq_while_identity_3%
!model_6_salt_seq_while_identity_4%
!model_6_salt_seq_while_identity_5;
7model_6_salt_seq_while_model_6_salt_seq_strided_slice_1w
smodel_6_salt_seq_while_tensorarrayv2read_tensorlistgetitem_model_6_salt_seq_tensorarrayunstack_tensorlistfromtensorU
Bmodel_6_salt_seq_while_lstm_cell_42_matmul_readvariableop_resource:	�W
Dmodel_6_salt_seq_while_lstm_cell_42_matmul_1_readvariableop_resource:	 �R
Cmodel_6_salt_seq_while_lstm_cell_42_biasadd_readvariableop_resource:	���:model_6/salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOp�9model_6/salt_seq/while/lstm_cell_42/MatMul/ReadVariableOp�;model_6/salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOp�
Hmodel_6/salt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
:model_6/salt_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemumodel_6_salt_seq_while_tensorarrayv2read_tensorlistgetitem_model_6_salt_seq_tensorarrayunstack_tensorlistfromtensor_0"model_6_salt_seq_while_placeholderQmodel_6/salt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
9model_6/salt_seq/while/lstm_cell_42/MatMul/ReadVariableOpReadVariableOpDmodel_6_salt_seq_while_lstm_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
*model_6/salt_seq/while/lstm_cell_42/MatMulMatMulAmodel_6/salt_seq/while/TensorArrayV2Read/TensorListGetItem:item:0Amodel_6/salt_seq/while/lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
;model_6/salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOpFmodel_6_salt_seq_while_lstm_cell_42_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
,model_6/salt_seq/while/lstm_cell_42/MatMul_1MatMul$model_6_salt_seq_while_placeholder_2Cmodel_6/salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'model_6/salt_seq/while/lstm_cell_42/addAddV24model_6/salt_seq/while/lstm_cell_42/MatMul:product:06model_6/salt_seq/while/lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
:model_6/salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOpEmodel_6_salt_seq_while_lstm_cell_42_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
+model_6/salt_seq/while/lstm_cell_42/BiasAddBiasAdd+model_6/salt_seq/while/lstm_cell_42/add:z:0Bmodel_6/salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
3model_6/salt_seq/while/lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
)model_6/salt_seq/while/lstm_cell_42/splitSplit<model_6/salt_seq/while/lstm_cell_42/split/split_dim:output:04model_6/salt_seq/while/lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split�
+model_6/salt_seq/while/lstm_cell_42/SigmoidSigmoid2model_6/salt_seq/while/lstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� �
-model_6/salt_seq/while/lstm_cell_42/Sigmoid_1Sigmoid2model_6/salt_seq/while/lstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� �
'model_6/salt_seq/while/lstm_cell_42/mulMul1model_6/salt_seq/while/lstm_cell_42/Sigmoid_1:y:0$model_6_salt_seq_while_placeholder_3*
T0*'
_output_shapes
:��������� �
(model_6/salt_seq/while/lstm_cell_42/ReluRelu2model_6/salt_seq/while/lstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
)model_6/salt_seq/while/lstm_cell_42/mul_1Mul/model_6/salt_seq/while/lstm_cell_42/Sigmoid:y:06model_6/salt_seq/while/lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� �
)model_6/salt_seq/while/lstm_cell_42/add_1AddV2+model_6/salt_seq/while/lstm_cell_42/mul:z:0-model_6/salt_seq/while/lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� �
-model_6/salt_seq/while/lstm_cell_42/Sigmoid_2Sigmoid2model_6/salt_seq/while/lstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� �
*model_6/salt_seq/while/lstm_cell_42/Relu_1Relu-model_6/salt_seq/while/lstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
)model_6/salt_seq/while/lstm_cell_42/mul_2Mul1model_6/salt_seq/while/lstm_cell_42/Sigmoid_2:y:08model_6/salt_seq/while/lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
;model_6/salt_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$model_6_salt_seq_while_placeholder_1"model_6_salt_seq_while_placeholder-model_6/salt_seq/while/lstm_cell_42/mul_2:z:0*
_output_shapes
: *
element_dtype0:���^
model_6/salt_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
model_6/salt_seq/while/addAddV2"model_6_salt_seq_while_placeholder%model_6/salt_seq/while/add/y:output:0*
T0*
_output_shapes
: `
model_6/salt_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
model_6/salt_seq/while/add_1AddV2:model_6_salt_seq_while_model_6_salt_seq_while_loop_counter'model_6/salt_seq/while/add_1/y:output:0*
T0*
_output_shapes
: �
model_6/salt_seq/while/IdentityIdentity model_6/salt_seq/while/add_1:z:0^model_6/salt_seq/while/NoOp*
T0*
_output_shapes
: �
!model_6/salt_seq/while/Identity_1Identity@model_6_salt_seq_while_model_6_salt_seq_while_maximum_iterations^model_6/salt_seq/while/NoOp*
T0*
_output_shapes
: �
!model_6/salt_seq/while/Identity_2Identitymodel_6/salt_seq/while/add:z:0^model_6/salt_seq/while/NoOp*
T0*
_output_shapes
: �
!model_6/salt_seq/while/Identity_3IdentityKmodel_6/salt_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model_6/salt_seq/while/NoOp*
T0*
_output_shapes
: :����
!model_6/salt_seq/while/Identity_4Identity-model_6/salt_seq/while/lstm_cell_42/mul_2:z:0^model_6/salt_seq/while/NoOp*
T0*'
_output_shapes
:��������� �
!model_6/salt_seq/while/Identity_5Identity-model_6/salt_seq/while/lstm_cell_42/add_1:z:0^model_6/salt_seq/while/NoOp*
T0*'
_output_shapes
:��������� �
model_6/salt_seq/while/NoOpNoOp;^model_6/salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOp:^model_6/salt_seq/while/lstm_cell_42/MatMul/ReadVariableOp<^model_6/salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "K
model_6_salt_seq_while_identity(model_6/salt_seq/while/Identity:output:0"O
!model_6_salt_seq_while_identity_1*model_6/salt_seq/while/Identity_1:output:0"O
!model_6_salt_seq_while_identity_2*model_6/salt_seq/while/Identity_2:output:0"O
!model_6_salt_seq_while_identity_3*model_6/salt_seq/while/Identity_3:output:0"O
!model_6_salt_seq_while_identity_4*model_6/salt_seq/while/Identity_4:output:0"O
!model_6_salt_seq_while_identity_5*model_6/salt_seq/while/Identity_5:output:0"�
Cmodel_6_salt_seq_while_lstm_cell_42_biasadd_readvariableop_resourceEmodel_6_salt_seq_while_lstm_cell_42_biasadd_readvariableop_resource_0"�
Dmodel_6_salt_seq_while_lstm_cell_42_matmul_1_readvariableop_resourceFmodel_6_salt_seq_while_lstm_cell_42_matmul_1_readvariableop_resource_0"�
Bmodel_6_salt_seq_while_lstm_cell_42_matmul_readvariableop_resourceDmodel_6_salt_seq_while_lstm_cell_42_matmul_readvariableop_resource_0"t
7model_6_salt_seq_while_model_6_salt_seq_strided_slice_19model_6_salt_seq_while_model_6_salt_seq_strided_slice_1_0"�
smodel_6_salt_seq_while_tensorarrayv2read_tensorlistgetitem_model_6_salt_seq_tensorarrayunstack_tensorlistfromtensorumodel_6_salt_seq_while_tensorarrayv2read_tensorlistgetitem_model_6_salt_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2x
:model_6/salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOp:model_6/salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOp2v
9model_6/salt_seq/while/lstm_cell_42/MatMul/ReadVariableOp9model_6/salt_seq/while/lstm_cell_42/MatMul/ReadVariableOp2z
;model_6/salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOp;model_6/salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_487146
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_487146___redundant_placeholder04
0while_while_cond_487146___redundant_placeholder14
0while_while_cond_487146___redundant_placeholder24
0while_while_cond_487146___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�"
�
while_body_485193
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_43_485217_0:	�.
while_lstm_cell_43_485219_0:	 �*
while_lstm_cell_43_485221_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_43_485217:	�,
while_lstm_cell_43_485219:	 �(
while_lstm_cell_43_485221:	���*while/lstm_cell_43/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
*while/lstm_cell_43/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_43_485217_0while_lstm_cell_43_485219_0while_lstm_cell_43_485221_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_43_layer_call_and_return_conditional_losses_485179�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_43/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :����
while/Identity_4Identity3while/lstm_cell_43/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_5Identity3while/lstm_cell_43/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:��������� y

while/NoOpNoOp+^while/lstm_cell_43/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_43_485217while_lstm_cell_43_485217_0"8
while_lstm_cell_43_485219while_lstm_cell_43_485219_0"8
while_lstm_cell_43_485221while_lstm_cell_43_485221_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2X
*while/lstm_cell_43/StatefulPartitionedCall*while/lstm_cell_43/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_qty_seq_layer_call_fn_487693

inputs
unknown:	�
	unknown_0:	 �
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_485613o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
E__inference_salt_pred_layer_call_and_return_conditional_losses_488362

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�8
�
while_body_485679
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_42_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_42_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_42_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_42_matmul_readvariableop_resource:	�F
3while_lstm_cell_42_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_42_biasadd_readvariableop_resource:	���)while/lstm_cell_42/BiasAdd/ReadVariableOp�(while/lstm_cell_42/MatMul/ReadVariableOp�*while/lstm_cell_42/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_42/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_42/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_42_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
while/lstm_cell_42/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_42/addAddV2#while/lstm_cell_42/MatMul:product:0%while/lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_42_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_42/BiasAddBiasAddwhile/lstm_cell_42/add:z:01while/lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_42/splitSplit+while/lstm_cell_42/split/split_dim:output:0#while/lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitz
while/lstm_cell_42/SigmoidSigmoid!while/lstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_42/Sigmoid_1Sigmoid!while/lstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mulMul while/lstm_cell_42/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� t
while/lstm_cell_42/ReluRelu!while/lstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mul_1Mulwhile/lstm_cell_42/Sigmoid:y:0%while/lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/add_1AddV2while/lstm_cell_42/mul:z:0while/lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_42/Sigmoid_2Sigmoid!while/lstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� q
while/lstm_cell_42/Relu_1Reluwhile/lstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mul_2Mul while/lstm_cell_42/Sigmoid_2:y:0'while/lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_42/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���y
while/Identity_4Identitywhile/lstm_cell_42/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� y
while/Identity_5Identitywhile/lstm_cell_42/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp*^while/lstm_cell_42/BiasAdd/ReadVariableOp)^while/lstm_cell_42/MatMul/ReadVariableOp+^while/lstm_cell_42/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_42_biasadd_readvariableop_resource4while_lstm_cell_42_biasadd_readvariableop_resource_0"l
3while_lstm_cell_42_matmul_1_readvariableop_resource5while_lstm_cell_42_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_42_matmul_readvariableop_resource3while_lstm_cell_42_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_42/BiasAdd/ReadVariableOp)while/lstm_cell_42/BiasAdd/ReadVariableOp2T
(while/lstm_cell_42/MatMul/ReadVariableOp(while/lstm_cell_42/MatMul/ReadVariableOp2X
*while/lstm_cell_42/MatMul_1/ReadVariableOp*while/lstm_cell_42/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
c
*__inference_qty_seq_2_layer_call_fn_488313

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_485867o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�8
�
while_body_487576
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_42_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_42_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_42_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_42_matmul_readvariableop_resource:	�F
3while_lstm_cell_42_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_42_biasadd_readvariableop_resource:	���)while/lstm_cell_42/BiasAdd/ReadVariableOp�(while/lstm_cell_42/MatMul/ReadVariableOp�*while/lstm_cell_42/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_42/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_42/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_42_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
while/lstm_cell_42/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_42/addAddV2#while/lstm_cell_42/MatMul:product:0%while/lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_42_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_42/BiasAddBiasAddwhile/lstm_cell_42/add:z:01while/lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_42/splitSplit+while/lstm_cell_42/split/split_dim:output:0#while/lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitz
while/lstm_cell_42/SigmoidSigmoid!while/lstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_42/Sigmoid_1Sigmoid!while/lstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mulMul while/lstm_cell_42/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� t
while/lstm_cell_42/ReluRelu!while/lstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mul_1Mulwhile/lstm_cell_42/Sigmoid:y:0%while/lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/add_1AddV2while/lstm_cell_42/mul:z:0while/lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_42/Sigmoid_2Sigmoid!while/lstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� q
while/lstm_cell_42/Relu_1Reluwhile/lstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mul_2Mul while/lstm_cell_42/Sigmoid_2:y:0'while/lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_42/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���y
while/Identity_4Identitywhile/lstm_cell_42/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� y
while/Identity_5Identitywhile/lstm_cell_42/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp*^while/lstm_cell_42/BiasAdd/ReadVariableOp)^while/lstm_cell_42/MatMul/ReadVariableOp+^while/lstm_cell_42/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_42_biasadd_readvariableop_resource4while_lstm_cell_42_biasadd_readvariableop_resource_0"l
3while_lstm_cell_42_matmul_1_readvariableop_resource5while_lstm_cell_42_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_42_matmul_readvariableop_resource3while_lstm_cell_42_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_42/BiasAdd/ReadVariableOp)while/lstm_cell_42/BiasAdd/ReadVariableOp2T
(while/lstm_cell_42/MatMul/ReadVariableOp(while/lstm_cell_42/MatMul/ReadVariableOp2X
*while/lstm_cell_42/MatMul_1/ReadVariableOp*while/lstm_cell_42/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_qty_seq_layer_call_fn_487704

inputs
unknown:	�
	unknown_0:	 �
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_486214o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�J
�
C__inference_qty_seq_layer_call_and_return_conditional_losses_486214

inputs>
+lstm_cell_43_matmul_readvariableop_resource:	�@
-lstm_cell_43_matmul_1_readvariableop_resource:	 �;
,lstm_cell_43_biasadd_readvariableop_resource:	�
identity��#lstm_cell_43/BiasAdd/ReadVariableOp�"lstm_cell_43/MatMul/ReadVariableOp�$lstm_cell_43/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_43/MatMul/ReadVariableOpReadVariableOp+lstm_cell_43_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_43/MatMulMatMulstrided_slice_2:output:0*lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_43_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
lstm_cell_43/MatMul_1MatMulzeros:output:0,lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_43/addAddV2lstm_cell_43/MatMul:product:0lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_43_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_43/BiasAddBiasAddlstm_cell_43/add:z:0+lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_43/splitSplit%lstm_cell_43/split/split_dim:output:0lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitn
lstm_cell_43/SigmoidSigmoidlstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� p
lstm_cell_43/Sigmoid_1Sigmoidlstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� w
lstm_cell_43/mulMullstm_cell_43/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� h
lstm_cell_43/ReluRelulstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
lstm_cell_43/mul_1Mullstm_cell_43/Sigmoid:y:0lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� {
lstm_cell_43/add_1AddV2lstm_cell_43/mul:z:0lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� p
lstm_cell_43/Sigmoid_2Sigmoidlstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� e
lstm_cell_43/Relu_1Relulstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
lstm_cell_43/mul_2Mullstm_cell_43/Sigmoid_2:y:0!lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_43_matmul_readvariableop_resource-lstm_cell_43_matmul_1_readvariableop_resource,lstm_cell_43_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_486130*
condR
while_cond_486129*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp$^lstm_cell_43/BiasAdd/ReadVariableOp#^lstm_cell_43/MatMul/ReadVariableOp%^lstm_cell_43/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_43/BiasAdd/ReadVariableOp#lstm_cell_43/BiasAdd/ReadVariableOp2H
"lstm_cell_43/MatMul/ReadVariableOp"lstm_cell_43/MatMul/ReadVariableOp2L
$lstm_cell_43/MatMul_1/ReadVariableOp$lstm_cell_43/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
+__inference_salt_seq_2_layer_call_fn_488286

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_485890o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
H__inference_lstm_cell_42_layer_call_and_return_conditional_losses_488428

inputs
states_0
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	 �.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:��������� V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:��������� U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:��������� N
ReluRelusplit:output:2*
T0*'
_output_shapes
:��������� _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:��������� T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:��������� V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:��������� K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:��������� c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:��������� :��������� : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/1
�
�
"__inference__traced_restore_488778
file_prefix3
!assignvariableop_salt_pred_kernel:@/
!assignvariableop_1_salt_pred_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: B
/assignvariableop_7_salt_seq_lstm_cell_42_kernel:	�L
9assignvariableop_8_salt_seq_lstm_cell_42_recurrent_kernel:	 �<
-assignvariableop_9_salt_seq_lstm_cell_42_bias:	�B
/assignvariableop_10_qty_seq_lstm_cell_43_kernel:	�L
9assignvariableop_11_qty_seq_lstm_cell_43_recurrent_kernel:	 �<
-assignvariableop_12_qty_seq_lstm_cell_43_bias:	�#
assignvariableop_13_total: #
assignvariableop_14_count: =
+assignvariableop_15_adam_salt_pred_kernel_m:@7
)assignvariableop_16_adam_salt_pred_bias_m:J
7assignvariableop_17_adam_salt_seq_lstm_cell_42_kernel_m:	�T
Aassignvariableop_18_adam_salt_seq_lstm_cell_42_recurrent_kernel_m:	 �D
5assignvariableop_19_adam_salt_seq_lstm_cell_42_bias_m:	�I
6assignvariableop_20_adam_qty_seq_lstm_cell_43_kernel_m:	�S
@assignvariableop_21_adam_qty_seq_lstm_cell_43_recurrent_kernel_m:	 �C
4assignvariableop_22_adam_qty_seq_lstm_cell_43_bias_m:	�=
+assignvariableop_23_adam_salt_pred_kernel_v:@7
)assignvariableop_24_adam_salt_pred_bias_v:J
7assignvariableop_25_adam_salt_seq_lstm_cell_42_kernel_v:	�T
Aassignvariableop_26_adam_salt_seq_lstm_cell_42_recurrent_kernel_v:	 �D
5assignvariableop_27_adam_salt_seq_lstm_cell_42_bias_v:	�I
6assignvariableop_28_adam_qty_seq_lstm_cell_43_kernel_v:	�S
@assignvariableop_29_adam_qty_seq_lstm_cell_43_recurrent_kernel_v:	 �C
4assignvariableop_30_adam_qty_seq_lstm_cell_43_bias_v:	�
identity_32��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp!assignvariableop_salt_pred_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_salt_pred_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp/assignvariableop_7_salt_seq_lstm_cell_42_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp9assignvariableop_8_salt_seq_lstm_cell_42_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp-assignvariableop_9_salt_seq_lstm_cell_42_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_qty_seq_lstm_cell_43_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_qty_seq_lstm_cell_43_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp-assignvariableop_12_qty_seq_lstm_cell_43_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_salt_pred_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_salt_pred_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp7assignvariableop_17_adam_salt_seq_lstm_cell_42_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpAassignvariableop_18_adam_salt_seq_lstm_cell_42_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp5assignvariableop_19_adam_salt_seq_lstm_cell_42_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_qty_seq_lstm_cell_43_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_qty_seq_lstm_cell_43_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_qty_seq_lstm_cell_43_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_salt_pred_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_salt_pred_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp7assignvariableop_25_adam_salt_seq_lstm_cell_42_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpAassignvariableop_26_adam_salt_seq_lstm_cell_42_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp5assignvariableop_27_adam_salt_seq_lstm_cell_42_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_qty_seq_lstm_cell_43_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_qty_seq_lstm_cell_43_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_qty_seq_lstm_cell_43_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
G
+__inference_salt_seq_2_layer_call_fn_488281

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_485776`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
qty_seq_while_cond_486772,
(qty_seq_while_qty_seq_while_loop_counter2
.qty_seq_while_qty_seq_while_maximum_iterations
qty_seq_while_placeholder
qty_seq_while_placeholder_1
qty_seq_while_placeholder_2
qty_seq_while_placeholder_3.
*qty_seq_while_less_qty_seq_strided_slice_1D
@qty_seq_while_qty_seq_while_cond_486772___redundant_placeholder0D
@qty_seq_while_qty_seq_while_cond_486772___redundant_placeholder1D
@qty_seq_while_qty_seq_while_cond_486772___redundant_placeholder2D
@qty_seq_while_qty_seq_while_cond_486772___redundant_placeholder3
qty_seq_while_identity
�
qty_seq/while/LessLessqty_seq_while_placeholder*qty_seq_while_less_qty_seq_strided_slice_1*
T0*
_output_shapes
: [
qty_seq/while/IdentityIdentityqty_seq/while/Less:z:0*
T0
*
_output_shapes
: "9
qty_seq_while_identityqty_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�8
�
while_body_487433
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_42_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_42_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_42_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_42_matmul_readvariableop_resource:	�F
3while_lstm_cell_42_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_42_biasadd_readvariableop_resource:	���)while/lstm_cell_42/BiasAdd/ReadVariableOp�(while/lstm_cell_42/MatMul/ReadVariableOp�*while/lstm_cell_42/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_42/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_42/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_42_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
while/lstm_cell_42/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_42/addAddV2#while/lstm_cell_42/MatMul:product:0%while/lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_42_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_42/BiasAddBiasAddwhile/lstm_cell_42/add:z:01while/lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_42/splitSplit+while/lstm_cell_42/split/split_dim:output:0#while/lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitz
while/lstm_cell_42/SigmoidSigmoid!while/lstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_42/Sigmoid_1Sigmoid!while/lstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mulMul while/lstm_cell_42/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� t
while/lstm_cell_42/ReluRelu!while/lstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mul_1Mulwhile/lstm_cell_42/Sigmoid:y:0%while/lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/add_1AddV2while/lstm_cell_42/mul:z:0while/lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_42/Sigmoid_2Sigmoid!while/lstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� q
while/lstm_cell_42/Relu_1Reluwhile/lstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mul_2Mul while/lstm_cell_42/Sigmoid_2:y:0'while/lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_42/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���y
while/Identity_4Identitywhile/lstm_cell_42/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� y
while/Identity_5Identitywhile/lstm_cell_42/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp*^while/lstm_cell_42/BiasAdd/ReadVariableOp)^while/lstm_cell_42/MatMul/ReadVariableOp+^while/lstm_cell_42/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_42_biasadd_readvariableop_resource4while_lstm_cell_42_biasadd_readvariableop_resource_0"l
3while_lstm_cell_42_matmul_1_readvariableop_resource5while_lstm_cell_42_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_42_matmul_readvariableop_resource3while_lstm_cell_42_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_42/BiasAdd/ReadVariableOp)while/lstm_cell_42/BiasAdd/ReadVariableOp2T
(while/lstm_cell_42/MatMul/ReadVariableOp(while/lstm_cell_42/MatMul/ReadVariableOp2X
*while/lstm_cell_42/MatMul_1/ReadVariableOp*while/lstm_cell_42/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�"
�
while_body_484843
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_42_484867_0:	�.
while_lstm_cell_42_484869_0:	 �*
while_lstm_cell_42_484871_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_42_484867:	�,
while_lstm_cell_42_484869:	 �(
while_lstm_cell_42_484871:	���*while/lstm_cell_42/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
*while/lstm_cell_42/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_42_484867_0while_lstm_cell_42_484869_0while_lstm_cell_42_484871_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_42_layer_call_and_return_conditional_losses_484829�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_42/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :����
while/Identity_4Identity3while/lstm_cell_42/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_5Identity3while/lstm_cell_42/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:��������� y

while/NoOpNoOp+^while/lstm_cell_42/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_42_484867while_lstm_cell_42_484867_0"8
while_lstm_cell_42_484869while_lstm_cell_42_484869_0"8
while_lstm_cell_42_484871while_lstm_cell_42_484871_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2X
*while/lstm_cell_42/StatefulPartitionedCall*while/lstm_cell_42/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
)__inference_salt_seq_layer_call_fn_487066
inputs_0
unknown:	�
	unknown_0:	 �
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_485103o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
!model_6_qty_seq_while_cond_484528<
8model_6_qty_seq_while_model_6_qty_seq_while_loop_counterB
>model_6_qty_seq_while_model_6_qty_seq_while_maximum_iterations%
!model_6_qty_seq_while_placeholder'
#model_6_qty_seq_while_placeholder_1'
#model_6_qty_seq_while_placeholder_2'
#model_6_qty_seq_while_placeholder_3>
:model_6_qty_seq_while_less_model_6_qty_seq_strided_slice_1T
Pmodel_6_qty_seq_while_model_6_qty_seq_while_cond_484528___redundant_placeholder0T
Pmodel_6_qty_seq_while_model_6_qty_seq_while_cond_484528___redundant_placeholder1T
Pmodel_6_qty_seq_while_model_6_qty_seq_while_cond_484528___redundant_placeholder2T
Pmodel_6_qty_seq_while_model_6_qty_seq_while_cond_484528___redundant_placeholder3"
model_6_qty_seq_while_identity
�
model_6/qty_seq/while/LessLess!model_6_qty_seq_while_placeholder:model_6_qty_seq_while_less_model_6_qty_seq_strided_slice_1*
T0*
_output_shapes
: k
model_6/qty_seq/while/IdentityIdentitymodel_6/qty_seq/while/Less:z:0*
T0
*
_output_shapes
: "I
model_6_qty_seq_while_identity'model_6/qty_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
c
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_488318

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
H__inference_lstm_cell_43_layer_call_and_return_conditional_losses_485179

inputs

states
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	 �.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:��������� V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:��������� U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:��������� N
ReluRelusplit:output:2*
T0*'
_output_shapes
:��������� _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:��������� T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:��������� V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:��������� K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:��������� c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:��������� :��������� : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates
�
�
H__inference_lstm_cell_43_layer_call_and_return_conditional_losses_488558

inputs
states_0
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	 �.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:��������� V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:��������� U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:��������� N
ReluRelusplit:output:2*
T0*'
_output_shapes
:��������� _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:��������� T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:��������� V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:��������� K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:��������� c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:��������� :��������� : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/1
�
�
(__inference_qty_seq_layer_call_fn_487671
inputs_0
unknown:	�
	unknown_0:	 �
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_485262o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
��
�	
!__inference__wrapped_model_484762
	salt_data
quantity_dataN
;model_6_qty_seq_lstm_cell_43_matmul_readvariableop_resource:	�P
=model_6_qty_seq_lstm_cell_43_matmul_1_readvariableop_resource:	 �K
<model_6_qty_seq_lstm_cell_43_biasadd_readvariableop_resource:	�O
<model_6_salt_seq_lstm_cell_42_matmul_readvariableop_resource:	�Q
>model_6_salt_seq_lstm_cell_42_matmul_1_readvariableop_resource:	 �L
=model_6_salt_seq_lstm_cell_42_biasadd_readvariableop_resource:	�B
0model_6_salt_pred_matmul_readvariableop_resource:@?
1model_6_salt_pred_biasadd_readvariableop_resource:
identity��3model_6/qty_seq/lstm_cell_43/BiasAdd/ReadVariableOp�2model_6/qty_seq/lstm_cell_43/MatMul/ReadVariableOp�4model_6/qty_seq/lstm_cell_43/MatMul_1/ReadVariableOp�model_6/qty_seq/while�(model_6/salt_pred/BiasAdd/ReadVariableOp�'model_6/salt_pred/MatMul/ReadVariableOp�4model_6/salt_seq/lstm_cell_42/BiasAdd/ReadVariableOp�3model_6/salt_seq/lstm_cell_42/MatMul/ReadVariableOp�5model_6/salt_seq/lstm_cell_42/MatMul_1/ReadVariableOp�model_6/salt_seq/whileR
model_6/qty_seq/ShapeShapequantity_data*
T0*
_output_shapes
:m
#model_6/qty_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model_6/qty_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model_6/qty_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model_6/qty_seq/strided_sliceStridedSlicemodel_6/qty_seq/Shape:output:0,model_6/qty_seq/strided_slice/stack:output:0.model_6/qty_seq/strided_slice/stack_1:output:0.model_6/qty_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
model_6/qty_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
model_6/qty_seq/zeros/packedPack&model_6/qty_seq/strided_slice:output:0'model_6/qty_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
model_6/qty_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model_6/qty_seq/zerosFill%model_6/qty_seq/zeros/packed:output:0$model_6/qty_seq/zeros/Const:output:0*
T0*'
_output_shapes
:��������� b
 model_6/qty_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
model_6/qty_seq/zeros_1/packedPack&model_6/qty_seq/strided_slice:output:0)model_6/qty_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
model_6/qty_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model_6/qty_seq/zeros_1Fill'model_6/qty_seq/zeros_1/packed:output:0&model_6/qty_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� s
model_6/qty_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
model_6/qty_seq/transpose	Transposequantity_data'model_6/qty_seq/transpose/perm:output:0*
T0*+
_output_shapes
:���������d
model_6/qty_seq/Shape_1Shapemodel_6/qty_seq/transpose:y:0*
T0*
_output_shapes
:o
%model_6/qty_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model_6/qty_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model_6/qty_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model_6/qty_seq/strided_slice_1StridedSlice model_6/qty_seq/Shape_1:output:0.model_6/qty_seq/strided_slice_1/stack:output:00model_6/qty_seq/strided_slice_1/stack_1:output:00model_6/qty_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+model_6/qty_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
model_6/qty_seq/TensorArrayV2TensorListReserve4model_6/qty_seq/TensorArrayV2/element_shape:output:0(model_6/qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Emodel_6/qty_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
7model_6/qty_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_6/qty_seq/transpose:y:0Nmodel_6/qty_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���o
%model_6/qty_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model_6/qty_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model_6/qty_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model_6/qty_seq/strided_slice_2StridedSlicemodel_6/qty_seq/transpose:y:0.model_6/qty_seq/strided_slice_2/stack:output:00model_6/qty_seq/strided_slice_2/stack_1:output:00model_6/qty_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
2model_6/qty_seq/lstm_cell_43/MatMul/ReadVariableOpReadVariableOp;model_6_qty_seq_lstm_cell_43_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
#model_6/qty_seq/lstm_cell_43/MatMulMatMul(model_6/qty_seq/strided_slice_2:output:0:model_6/qty_seq/lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
4model_6/qty_seq/lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOp=model_6_qty_seq_lstm_cell_43_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
%model_6/qty_seq/lstm_cell_43/MatMul_1MatMulmodel_6/qty_seq/zeros:output:0<model_6/qty_seq/lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 model_6/qty_seq/lstm_cell_43/addAddV2-model_6/qty_seq/lstm_cell_43/MatMul:product:0/model_6/qty_seq/lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
3model_6/qty_seq/lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOp<model_6_qty_seq_lstm_cell_43_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$model_6/qty_seq/lstm_cell_43/BiasAddBiasAdd$model_6/qty_seq/lstm_cell_43/add:z:0;model_6/qty_seq/lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������n
,model_6/qty_seq/lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
"model_6/qty_seq/lstm_cell_43/splitSplit5model_6/qty_seq/lstm_cell_43/split/split_dim:output:0-model_6/qty_seq/lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split�
$model_6/qty_seq/lstm_cell_43/SigmoidSigmoid+model_6/qty_seq/lstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� �
&model_6/qty_seq/lstm_cell_43/Sigmoid_1Sigmoid+model_6/qty_seq/lstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� �
 model_6/qty_seq/lstm_cell_43/mulMul*model_6/qty_seq/lstm_cell_43/Sigmoid_1:y:0 model_6/qty_seq/zeros_1:output:0*
T0*'
_output_shapes
:��������� �
!model_6/qty_seq/lstm_cell_43/ReluRelu+model_6/qty_seq/lstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
"model_6/qty_seq/lstm_cell_43/mul_1Mul(model_6/qty_seq/lstm_cell_43/Sigmoid:y:0/model_6/qty_seq/lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� �
"model_6/qty_seq/lstm_cell_43/add_1AddV2$model_6/qty_seq/lstm_cell_43/mul:z:0&model_6/qty_seq/lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� �
&model_6/qty_seq/lstm_cell_43/Sigmoid_2Sigmoid+model_6/qty_seq/lstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� �
#model_6/qty_seq/lstm_cell_43/Relu_1Relu&model_6/qty_seq/lstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
"model_6/qty_seq/lstm_cell_43/mul_2Mul*model_6/qty_seq/lstm_cell_43/Sigmoid_2:y:01model_6/qty_seq/lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� ~
-model_6/qty_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
model_6/qty_seq/TensorArrayV2_1TensorListReserve6model_6/qty_seq/TensorArrayV2_1/element_shape:output:0(model_6/qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���V
model_6/qty_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(model_6/qty_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������d
"model_6/qty_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
model_6/qty_seq/whileWhile+model_6/qty_seq/while/loop_counter:output:01model_6/qty_seq/while/maximum_iterations:output:0model_6/qty_seq/time:output:0(model_6/qty_seq/TensorArrayV2_1:handle:0model_6/qty_seq/zeros:output:0 model_6/qty_seq/zeros_1:output:0(model_6/qty_seq/strided_slice_1:output:0Gmodel_6/qty_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:0;model_6_qty_seq_lstm_cell_43_matmul_readvariableop_resource=model_6_qty_seq_lstm_cell_43_matmul_1_readvariableop_resource<model_6_qty_seq_lstm_cell_43_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *-
body%R#
!model_6_qty_seq_while_body_484529*-
cond%R#
!model_6_qty_seq_while_cond_484528*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
@model_6/qty_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
2model_6/qty_seq/TensorArrayV2Stack/TensorListStackTensorListStackmodel_6/qty_seq/while:output:3Imodel_6/qty_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0x
%model_6/qty_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������q
'model_6/qty_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'model_6/qty_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model_6/qty_seq/strided_slice_3StridedSlice;model_6/qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0.model_6/qty_seq/strided_slice_3/stack:output:00model_6/qty_seq/strided_slice_3/stack_1:output:00model_6/qty_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_masku
 model_6/qty_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
model_6/qty_seq/transpose_1	Transpose;model_6/qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0)model_6/qty_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� k
model_6/qty_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    O
model_6/salt_seq/ShapeShape	salt_data*
T0*
_output_shapes
:n
$model_6/salt_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&model_6/salt_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&model_6/salt_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model_6/salt_seq/strided_sliceStridedSlicemodel_6/salt_seq/Shape:output:0-model_6/salt_seq/strided_slice/stack:output:0/model_6/salt_seq/strided_slice/stack_1:output:0/model_6/salt_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model_6/salt_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
model_6/salt_seq/zeros/packedPack'model_6/salt_seq/strided_slice:output:0(model_6/salt_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
model_6/salt_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model_6/salt_seq/zerosFill&model_6/salt_seq/zeros/packed:output:0%model_6/salt_seq/zeros/Const:output:0*
T0*'
_output_shapes
:��������� c
!model_6/salt_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : �
model_6/salt_seq/zeros_1/packedPack'model_6/salt_seq/strided_slice:output:0*model_6/salt_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:c
model_6/salt_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model_6/salt_seq/zeros_1Fill(model_6/salt_seq/zeros_1/packed:output:0'model_6/salt_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� t
model_6/salt_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
model_6/salt_seq/transpose	Transpose	salt_data(model_6/salt_seq/transpose/perm:output:0*
T0*+
_output_shapes
:���������f
model_6/salt_seq/Shape_1Shapemodel_6/salt_seq/transpose:y:0*
T0*
_output_shapes
:p
&model_6/salt_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(model_6/salt_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(model_6/salt_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 model_6/salt_seq/strided_slice_1StridedSlice!model_6/salt_seq/Shape_1:output:0/model_6/salt_seq/strided_slice_1/stack:output:01model_6/salt_seq/strided_slice_1/stack_1:output:01model_6/salt_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,model_6/salt_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
model_6/salt_seq/TensorArrayV2TensorListReserve5model_6/salt_seq/TensorArrayV2/element_shape:output:0)model_6/salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Fmodel_6/salt_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
8model_6/salt_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_6/salt_seq/transpose:y:0Omodel_6/salt_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���p
&model_6/salt_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(model_6/salt_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(model_6/salt_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 model_6/salt_seq/strided_slice_2StridedSlicemodel_6/salt_seq/transpose:y:0/model_6/salt_seq/strided_slice_2/stack:output:01model_6/salt_seq/strided_slice_2/stack_1:output:01model_6/salt_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
3model_6/salt_seq/lstm_cell_42/MatMul/ReadVariableOpReadVariableOp<model_6_salt_seq_lstm_cell_42_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
$model_6/salt_seq/lstm_cell_42/MatMulMatMul)model_6/salt_seq/strided_slice_2:output:0;model_6/salt_seq/lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
5model_6/salt_seq/lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOp>model_6_salt_seq_lstm_cell_42_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
&model_6/salt_seq/lstm_cell_42/MatMul_1MatMulmodel_6/salt_seq/zeros:output:0=model_6/salt_seq/lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
!model_6/salt_seq/lstm_cell_42/addAddV2.model_6/salt_seq/lstm_cell_42/MatMul:product:00model_6/salt_seq/lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
4model_6/salt_seq/lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOp=model_6_salt_seq_lstm_cell_42_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%model_6/salt_seq/lstm_cell_42/BiasAddBiasAdd%model_6/salt_seq/lstm_cell_42/add:z:0<model_6/salt_seq/lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������o
-model_6/salt_seq/lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
#model_6/salt_seq/lstm_cell_42/splitSplit6model_6/salt_seq/lstm_cell_42/split/split_dim:output:0.model_6/salt_seq/lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split�
%model_6/salt_seq/lstm_cell_42/SigmoidSigmoid,model_6/salt_seq/lstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� �
'model_6/salt_seq/lstm_cell_42/Sigmoid_1Sigmoid,model_6/salt_seq/lstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� �
!model_6/salt_seq/lstm_cell_42/mulMul+model_6/salt_seq/lstm_cell_42/Sigmoid_1:y:0!model_6/salt_seq/zeros_1:output:0*
T0*'
_output_shapes
:��������� �
"model_6/salt_seq/lstm_cell_42/ReluRelu,model_6/salt_seq/lstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
#model_6/salt_seq/lstm_cell_42/mul_1Mul)model_6/salt_seq/lstm_cell_42/Sigmoid:y:00model_6/salt_seq/lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� �
#model_6/salt_seq/lstm_cell_42/add_1AddV2%model_6/salt_seq/lstm_cell_42/mul:z:0'model_6/salt_seq/lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� �
'model_6/salt_seq/lstm_cell_42/Sigmoid_2Sigmoid,model_6/salt_seq/lstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� �
$model_6/salt_seq/lstm_cell_42/Relu_1Relu'model_6/salt_seq/lstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
#model_6/salt_seq/lstm_cell_42/mul_2Mul+model_6/salt_seq/lstm_cell_42/Sigmoid_2:y:02model_6/salt_seq/lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 
.model_6/salt_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
 model_6/salt_seq/TensorArrayV2_1TensorListReserve7model_6/salt_seq/TensorArrayV2_1/element_shape:output:0)model_6/salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���W
model_6/salt_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : t
)model_6/salt_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������e
#model_6/salt_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
model_6/salt_seq/whileWhile,model_6/salt_seq/while/loop_counter:output:02model_6/salt_seq/while/maximum_iterations:output:0model_6/salt_seq/time:output:0)model_6/salt_seq/TensorArrayV2_1:handle:0model_6/salt_seq/zeros:output:0!model_6/salt_seq/zeros_1:output:0)model_6/salt_seq/strided_slice_1:output:0Hmodel_6/salt_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:0<model_6_salt_seq_lstm_cell_42_matmul_readvariableop_resource>model_6_salt_seq_lstm_cell_42_matmul_1_readvariableop_resource=model_6_salt_seq_lstm_cell_42_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *.
body&R$
"model_6_salt_seq_while_body_484668*.
cond&R$
"model_6_salt_seq_while_cond_484667*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
Amodel_6/salt_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
3model_6/salt_seq/TensorArrayV2Stack/TensorListStackTensorListStackmodel_6/salt_seq/while:output:3Jmodel_6/salt_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0y
&model_6/salt_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������r
(model_6/salt_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(model_6/salt_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 model_6/salt_seq/strided_slice_3StridedSlice<model_6/salt_seq/TensorArrayV2Stack/TensorListStack:tensor:0/model_6/salt_seq/strided_slice_3/stack:output:01model_6/salt_seq/strided_slice_3/stack_1:output:01model_6/salt_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maskv
!model_6/salt_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
model_6/salt_seq/transpose_1	Transpose<model_6/salt_seq/TensorArrayV2Stack/TensorListStack:tensor:0*model_6/salt_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� l
model_6/salt_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
model_6/salt_seq_2/IdentityIdentity)model_6/salt_seq/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� �
model_6/qty_seq_2/IdentityIdentity(model_6/qty_seq/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� ]
model_6/pattern/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_6/pattern/concatConcatV2$model_6/salt_seq_2/Identity:output:0#model_6/qty_seq_2/Identity:output:0$model_6/pattern/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������@�
'model_6/salt_pred/MatMul/ReadVariableOpReadVariableOp0model_6_salt_pred_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model_6/salt_pred/MatMulMatMulmodel_6/pattern/concat:output:0/model_6/salt_pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(model_6/salt_pred/BiasAdd/ReadVariableOpReadVariableOp1model_6_salt_pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_6/salt_pred/BiasAddBiasAdd"model_6/salt_pred/MatMul:product:00model_6/salt_pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������q
IdentityIdentity"model_6/salt_pred/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp4^model_6/qty_seq/lstm_cell_43/BiasAdd/ReadVariableOp3^model_6/qty_seq/lstm_cell_43/MatMul/ReadVariableOp5^model_6/qty_seq/lstm_cell_43/MatMul_1/ReadVariableOp^model_6/qty_seq/while)^model_6/salt_pred/BiasAdd/ReadVariableOp(^model_6/salt_pred/MatMul/ReadVariableOp5^model_6/salt_seq/lstm_cell_42/BiasAdd/ReadVariableOp4^model_6/salt_seq/lstm_cell_42/MatMul/ReadVariableOp6^model_6/salt_seq/lstm_cell_42/MatMul_1/ReadVariableOp^model_6/salt_seq/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2j
3model_6/qty_seq/lstm_cell_43/BiasAdd/ReadVariableOp3model_6/qty_seq/lstm_cell_43/BiasAdd/ReadVariableOp2h
2model_6/qty_seq/lstm_cell_43/MatMul/ReadVariableOp2model_6/qty_seq/lstm_cell_43/MatMul/ReadVariableOp2l
4model_6/qty_seq/lstm_cell_43/MatMul_1/ReadVariableOp4model_6/qty_seq/lstm_cell_43/MatMul_1/ReadVariableOp2.
model_6/qty_seq/whilemodel_6/qty_seq/while2T
(model_6/salt_pred/BiasAdd/ReadVariableOp(model_6/salt_pred/BiasAdd/ReadVariableOp2R
'model_6/salt_pred/MatMul/ReadVariableOp'model_6/salt_pred/MatMul/ReadVariableOp2l
4model_6/salt_seq/lstm_cell_42/BiasAdd/ReadVariableOp4model_6/salt_seq/lstm_cell_42/BiasAdd/ReadVariableOp2j
3model_6/salt_seq/lstm_cell_42/MatMul/ReadVariableOp3model_6/salt_seq/lstm_cell_42/MatMul/ReadVariableOp2n
5model_6/salt_seq/lstm_cell_42/MatMul_1/ReadVariableOp5model_6/salt_seq/lstm_cell_42/MatMul_1/ReadVariableOp20
model_6/salt_seq/whilemodel_6/salt_seq/while:V R
+
_output_shapes
:���������
#
_user_specified_name	salt_data:ZV
+
_output_shapes
:���������
'
_user_specified_namequantity_data
�8
�
D__inference_salt_seq_layer_call_and_return_conditional_losses_485103

inputs&
lstm_cell_42_485021:	�&
lstm_cell_42_485023:	 �"
lstm_cell_42_485025:	�
identity��$lstm_cell_42/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
$lstm_cell_42/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_42_485021lstm_cell_42_485023lstm_cell_42_485025*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_42_layer_call_and_return_conditional_losses_484975n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_42_485021lstm_cell_42_485023lstm_cell_42_485025*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_485034*
condR
while_cond_485033*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� u
NoOpNoOp%^lstm_cell_42/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_42/StatefulPartitionedCall$lstm_cell_42/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�

�
salt_seq_while_cond_486618.
*salt_seq_while_salt_seq_while_loop_counter4
0salt_seq_while_salt_seq_while_maximum_iterations
salt_seq_while_placeholder 
salt_seq_while_placeholder_1 
salt_seq_while_placeholder_2 
salt_seq_while_placeholder_30
,salt_seq_while_less_salt_seq_strided_slice_1F
Bsalt_seq_while_salt_seq_while_cond_486618___redundant_placeholder0F
Bsalt_seq_while_salt_seq_while_cond_486618___redundant_placeholder1F
Bsalt_seq_while_salt_seq_while_cond_486618___redundant_placeholder2F
Bsalt_seq_while_salt_seq_while_cond_486618___redundant_placeholder3
salt_seq_while_identity
�
salt_seq/while/LessLesssalt_seq_while_placeholder,salt_seq_while_less_salt_seq_strided_slice_1*
T0*
_output_shapes
: ]
salt_seq/while/IdentityIdentitysalt_seq/while/Less:z:0*
T0
*
_output_shapes
: ";
salt_seq_while_identity salt_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�

�
salt_seq_while_cond_486911.
*salt_seq_while_salt_seq_while_loop_counter4
0salt_seq_while_salt_seq_while_maximum_iterations
salt_seq_while_placeholder 
salt_seq_while_placeholder_1 
salt_seq_while_placeholder_2 
salt_seq_while_placeholder_30
,salt_seq_while_less_salt_seq_strided_slice_1F
Bsalt_seq_while_salt_seq_while_cond_486911___redundant_placeholder0F
Bsalt_seq_while_salt_seq_while_cond_486911___redundant_placeholder1F
Bsalt_seq_while_salt_seq_while_cond_486911___redundant_placeholder2F
Bsalt_seq_while_salt_seq_while_cond_486911___redundant_placeholder3
salt_seq_while_identity
�
salt_seq/while/LessLesssalt_seq_while_placeholder,salt_seq_while_less_salt_seq_strided_slice_1*
T0*
_output_shapes
: ]
salt_seq/while/IdentityIdentitysalt_seq/while/Less:z:0*
T0
*
_output_shapes
: ";
salt_seq_while_identity salt_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
while_cond_485383
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_485383___redundant_placeholder04
0while_while_cond_485383___redundant_placeholder14
0while_while_cond_485383___redundant_placeholder24
0while_while_cond_485383___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
o
C__inference_pattern_layer_call_and_return_conditional_losses_488343
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������@W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:��������� :��������� :Q M
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�
m
C__inference_pattern_layer_call_and_return_conditional_losses_485792

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������@W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:��������� :��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
T
(__inference_pattern_layer_call_fn_488336
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_pattern_layer_call_and_return_conditional_losses_485792`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:��������� :��������� :Q M
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�

�
$__inference_signature_wrapper_487044
quantity_data
	salt_data
unknown:	�
	unknown_0:	 �
	unknown_1:	�
	unknown_2:	�
	unknown_3:	 �
	unknown_4:	�
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	salt_dataquantity_dataunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_484762o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namequantity_data:VR
+
_output_shapes
:���������
#
_user_specified_name	salt_data
�A
�

qty_seq_while_body_486480,
(qty_seq_while_qty_seq_while_loop_counter2
.qty_seq_while_qty_seq_while_maximum_iterations
qty_seq_while_placeholder
qty_seq_while_placeholder_1
qty_seq_while_placeholder_2
qty_seq_while_placeholder_3+
'qty_seq_while_qty_seq_strided_slice_1_0g
cqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0N
;qty_seq_while_lstm_cell_43_matmul_readvariableop_resource_0:	�P
=qty_seq_while_lstm_cell_43_matmul_1_readvariableop_resource_0:	 �K
<qty_seq_while_lstm_cell_43_biasadd_readvariableop_resource_0:	�
qty_seq_while_identity
qty_seq_while_identity_1
qty_seq_while_identity_2
qty_seq_while_identity_3
qty_seq_while_identity_4
qty_seq_while_identity_5)
%qty_seq_while_qty_seq_strided_slice_1e
aqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensorL
9qty_seq_while_lstm_cell_43_matmul_readvariableop_resource:	�N
;qty_seq_while_lstm_cell_43_matmul_1_readvariableop_resource:	 �I
:qty_seq_while_lstm_cell_43_biasadd_readvariableop_resource:	���1qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOp�0qty_seq/while/lstm_cell_43/MatMul/ReadVariableOp�2qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOp�
?qty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
1qty_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemcqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0qty_seq_while_placeholderHqty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
0qty_seq/while/lstm_cell_43/MatMul/ReadVariableOpReadVariableOp;qty_seq_while_lstm_cell_43_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
!qty_seq/while/lstm_cell_43/MatMulMatMul8qty_seq/while/TensorArrayV2Read/TensorListGetItem:item:08qty_seq/while/lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOp=qty_seq_while_lstm_cell_43_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
#qty_seq/while/lstm_cell_43/MatMul_1MatMulqty_seq_while_placeholder_2:qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
qty_seq/while/lstm_cell_43/addAddV2+qty_seq/while/lstm_cell_43/MatMul:product:0-qty_seq/while/lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
1qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOp<qty_seq_while_lstm_cell_43_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
"qty_seq/while/lstm_cell_43/BiasAddBiasAdd"qty_seq/while/lstm_cell_43/add:z:09qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
*qty_seq/while/lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 qty_seq/while/lstm_cell_43/splitSplit3qty_seq/while/lstm_cell_43/split/split_dim:output:0+qty_seq/while/lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split�
"qty_seq/while/lstm_cell_43/SigmoidSigmoid)qty_seq/while/lstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� �
$qty_seq/while/lstm_cell_43/Sigmoid_1Sigmoid)qty_seq/while/lstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� �
qty_seq/while/lstm_cell_43/mulMul(qty_seq/while/lstm_cell_43/Sigmoid_1:y:0qty_seq_while_placeholder_3*
T0*'
_output_shapes
:��������� �
qty_seq/while/lstm_cell_43/ReluRelu)qty_seq/while/lstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
 qty_seq/while/lstm_cell_43/mul_1Mul&qty_seq/while/lstm_cell_43/Sigmoid:y:0-qty_seq/while/lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� �
 qty_seq/while/lstm_cell_43/add_1AddV2"qty_seq/while/lstm_cell_43/mul:z:0$qty_seq/while/lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� �
$qty_seq/while/lstm_cell_43/Sigmoid_2Sigmoid)qty_seq/while/lstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� �
!qty_seq/while/lstm_cell_43/Relu_1Relu$qty_seq/while/lstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
 qty_seq/while/lstm_cell_43/mul_2Mul(qty_seq/while/lstm_cell_43/Sigmoid_2:y:0/qty_seq/while/lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
2qty_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemqty_seq_while_placeholder_1qty_seq_while_placeholder$qty_seq/while/lstm_cell_43/mul_2:z:0*
_output_shapes
: *
element_dtype0:���U
qty_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
qty_seq/while/addAddV2qty_seq_while_placeholderqty_seq/while/add/y:output:0*
T0*
_output_shapes
: W
qty_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
qty_seq/while/add_1AddV2(qty_seq_while_qty_seq_while_loop_counterqty_seq/while/add_1/y:output:0*
T0*
_output_shapes
: q
qty_seq/while/IdentityIdentityqty_seq/while/add_1:z:0^qty_seq/while/NoOp*
T0*
_output_shapes
: �
qty_seq/while/Identity_1Identity.qty_seq_while_qty_seq_while_maximum_iterations^qty_seq/while/NoOp*
T0*
_output_shapes
: q
qty_seq/while/Identity_2Identityqty_seq/while/add:z:0^qty_seq/while/NoOp*
T0*
_output_shapes
: �
qty_seq/while/Identity_3IdentityBqty_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^qty_seq/while/NoOp*
T0*
_output_shapes
: :����
qty_seq/while/Identity_4Identity$qty_seq/while/lstm_cell_43/mul_2:z:0^qty_seq/while/NoOp*
T0*'
_output_shapes
:��������� �
qty_seq/while/Identity_5Identity$qty_seq/while/lstm_cell_43/add_1:z:0^qty_seq/while/NoOp*
T0*'
_output_shapes
:��������� �
qty_seq/while/NoOpNoOp2^qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOp1^qty_seq/while/lstm_cell_43/MatMul/ReadVariableOp3^qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
qty_seq_while_identityqty_seq/while/Identity:output:0"=
qty_seq_while_identity_1!qty_seq/while/Identity_1:output:0"=
qty_seq_while_identity_2!qty_seq/while/Identity_2:output:0"=
qty_seq_while_identity_3!qty_seq/while/Identity_3:output:0"=
qty_seq_while_identity_4!qty_seq/while/Identity_4:output:0"=
qty_seq_while_identity_5!qty_seq/while/Identity_5:output:0"z
:qty_seq_while_lstm_cell_43_biasadd_readvariableop_resource<qty_seq_while_lstm_cell_43_biasadd_readvariableop_resource_0"|
;qty_seq_while_lstm_cell_43_matmul_1_readvariableop_resource=qty_seq_while_lstm_cell_43_matmul_1_readvariableop_resource_0"x
9qty_seq_while_lstm_cell_43_matmul_readvariableop_resource;qty_seq_while_lstm_cell_43_matmul_readvariableop_resource_0"P
%qty_seq_while_qty_seq_strided_slice_1'qty_seq_while_qty_seq_strided_slice_1_0"�
aqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensorcqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2f
1qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOp1qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOp2d
0qty_seq/while/lstm_cell_43/MatMul/ReadVariableOp0qty_seq/while/lstm_cell_43/MatMul/ReadVariableOp2h
2qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOp2qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�8
�
C__inference_qty_seq_layer_call_and_return_conditional_losses_485262

inputs&
lstm_cell_43_485180:	�&
lstm_cell_43_485182:	 �"
lstm_cell_43_485184:	�
identity��$lstm_cell_43/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
$lstm_cell_43/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_43_485180lstm_cell_43_485182lstm_cell_43_485184*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_43_layer_call_and_return_conditional_losses_485179n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_43_485180lstm_cell_43_485182lstm_cell_43_485184*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_485193*
condR
while_cond_485192*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� u
NoOpNoOp%^lstm_cell_43/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_43/StatefulPartitionedCall$lstm_cell_43/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�

�
(__inference_model_6_layer_call_fn_486420
inputs_0
inputs_1
unknown:	�
	unknown_0:	 �
	unknown_1:	�
	unknown_2:	�
	unknown_3:	 �
	unknown_4:	�
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_6_layer_call_and_return_conditional_losses_486275o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs/1
�	
d
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_488330

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:��������� Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
while_cond_484842
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_484842___redundant_placeholder04
0while_while_cond_484842___redundant_placeholder14
0while_while_cond_484842___redundant_placeholder24
0while_while_cond_484842___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�8
�
while_body_486130
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_43_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_43_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_43_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_43_matmul_readvariableop_resource:	�F
3while_lstm_cell_43_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_43_biasadd_readvariableop_resource:	���)while/lstm_cell_43/BiasAdd/ReadVariableOp�(while/lstm_cell_43/MatMul/ReadVariableOp�*while/lstm_cell_43/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_43/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_43_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_43/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_43_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
while/lstm_cell_43/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_43/addAddV2#while/lstm_cell_43/MatMul:product:0%while/lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_43_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_43/BiasAddBiasAddwhile/lstm_cell_43/add:z:01while/lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_43/splitSplit+while/lstm_cell_43/split/split_dim:output:0#while/lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitz
while/lstm_cell_43/SigmoidSigmoid!while/lstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_43/Sigmoid_1Sigmoid!while/lstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mulMul while/lstm_cell_43/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� t
while/lstm_cell_43/ReluRelu!while/lstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mul_1Mulwhile/lstm_cell_43/Sigmoid:y:0%while/lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/add_1AddV2while/lstm_cell_43/mul:z:0while/lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_43/Sigmoid_2Sigmoid!while/lstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� q
while/lstm_cell_43/Relu_1Reluwhile/lstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mul_2Mul while/lstm_cell_43/Sigmoid_2:y:0'while/lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_43/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���y
while/Identity_4Identitywhile/lstm_cell_43/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� y
while/Identity_5Identitywhile/lstm_cell_43/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp*^while/lstm_cell_43/BiasAdd/ReadVariableOp)^while/lstm_cell_43/MatMul/ReadVariableOp+^while/lstm_cell_43/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_43_biasadd_readvariableop_resource4while_lstm_cell_43_biasadd_readvariableop_resource_0"l
3while_lstm_cell_43_matmul_1_readvariableop_resource5while_lstm_cell_43_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_43_matmul_readvariableop_resource3while_lstm_cell_43_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_43/BiasAdd/ReadVariableOp)while/lstm_cell_43/BiasAdd/ReadVariableOp2T
(while/lstm_cell_43/MatMul/ReadVariableOp(while/lstm_cell_43/MatMul/ReadVariableOp2X
*while/lstm_cell_43/MatMul_1/ReadVariableOp*while/lstm_cell_43/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�J
�
C__inference_qty_seq_layer_call_and_return_conditional_losses_485613

inputs>
+lstm_cell_43_matmul_readvariableop_resource:	�@
-lstm_cell_43_matmul_1_readvariableop_resource:	 �;
,lstm_cell_43_biasadd_readvariableop_resource:	�
identity��#lstm_cell_43/BiasAdd/ReadVariableOp�"lstm_cell_43/MatMul/ReadVariableOp�$lstm_cell_43/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_43/MatMul/ReadVariableOpReadVariableOp+lstm_cell_43_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_43/MatMulMatMulstrided_slice_2:output:0*lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_43_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
lstm_cell_43/MatMul_1MatMulzeros:output:0,lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_43/addAddV2lstm_cell_43/MatMul:product:0lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_43_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_43/BiasAddBiasAddlstm_cell_43/add:z:0+lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_43/splitSplit%lstm_cell_43/split/split_dim:output:0lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitn
lstm_cell_43/SigmoidSigmoidlstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� p
lstm_cell_43/Sigmoid_1Sigmoidlstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� w
lstm_cell_43/mulMullstm_cell_43/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� h
lstm_cell_43/ReluRelulstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
lstm_cell_43/mul_1Mullstm_cell_43/Sigmoid:y:0lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� {
lstm_cell_43/add_1AddV2lstm_cell_43/mul:z:0lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� p
lstm_cell_43/Sigmoid_2Sigmoidlstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� e
lstm_cell_43/Relu_1Relulstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
lstm_cell_43/mul_2Mullstm_cell_43/Sigmoid_2:y:0!lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_43_matmul_readvariableop_resource-lstm_cell_43_matmul_1_readvariableop_resource,lstm_cell_43_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_485529*
condR
while_cond_485528*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp$^lstm_cell_43/BiasAdd/ReadVariableOp#^lstm_cell_43/MatMul/ReadVariableOp%^lstm_cell_43/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_43/BiasAdd/ReadVariableOp#lstm_cell_43/BiasAdd/ReadVariableOp2H
"lstm_cell_43/MatMul/ReadVariableOp"lstm_cell_43/MatMul/ReadVariableOp2L
$lstm_cell_43/MatMul_1/ReadVariableOp$lstm_cell_43/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�A
�

qty_seq_while_body_486773,
(qty_seq_while_qty_seq_while_loop_counter2
.qty_seq_while_qty_seq_while_maximum_iterations
qty_seq_while_placeholder
qty_seq_while_placeholder_1
qty_seq_while_placeholder_2
qty_seq_while_placeholder_3+
'qty_seq_while_qty_seq_strided_slice_1_0g
cqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0N
;qty_seq_while_lstm_cell_43_matmul_readvariableop_resource_0:	�P
=qty_seq_while_lstm_cell_43_matmul_1_readvariableop_resource_0:	 �K
<qty_seq_while_lstm_cell_43_biasadd_readvariableop_resource_0:	�
qty_seq_while_identity
qty_seq_while_identity_1
qty_seq_while_identity_2
qty_seq_while_identity_3
qty_seq_while_identity_4
qty_seq_while_identity_5)
%qty_seq_while_qty_seq_strided_slice_1e
aqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensorL
9qty_seq_while_lstm_cell_43_matmul_readvariableop_resource:	�N
;qty_seq_while_lstm_cell_43_matmul_1_readvariableop_resource:	 �I
:qty_seq_while_lstm_cell_43_biasadd_readvariableop_resource:	���1qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOp�0qty_seq/while/lstm_cell_43/MatMul/ReadVariableOp�2qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOp�
?qty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
1qty_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemcqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0qty_seq_while_placeholderHqty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
0qty_seq/while/lstm_cell_43/MatMul/ReadVariableOpReadVariableOp;qty_seq_while_lstm_cell_43_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
!qty_seq/while/lstm_cell_43/MatMulMatMul8qty_seq/while/TensorArrayV2Read/TensorListGetItem:item:08qty_seq/while/lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOp=qty_seq_while_lstm_cell_43_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
#qty_seq/while/lstm_cell_43/MatMul_1MatMulqty_seq_while_placeholder_2:qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
qty_seq/while/lstm_cell_43/addAddV2+qty_seq/while/lstm_cell_43/MatMul:product:0-qty_seq/while/lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
1qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOp<qty_seq_while_lstm_cell_43_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
"qty_seq/while/lstm_cell_43/BiasAddBiasAdd"qty_seq/while/lstm_cell_43/add:z:09qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
*qty_seq/while/lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 qty_seq/while/lstm_cell_43/splitSplit3qty_seq/while/lstm_cell_43/split/split_dim:output:0+qty_seq/while/lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split�
"qty_seq/while/lstm_cell_43/SigmoidSigmoid)qty_seq/while/lstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� �
$qty_seq/while/lstm_cell_43/Sigmoid_1Sigmoid)qty_seq/while/lstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� �
qty_seq/while/lstm_cell_43/mulMul(qty_seq/while/lstm_cell_43/Sigmoid_1:y:0qty_seq_while_placeholder_3*
T0*'
_output_shapes
:��������� �
qty_seq/while/lstm_cell_43/ReluRelu)qty_seq/while/lstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
 qty_seq/while/lstm_cell_43/mul_1Mul&qty_seq/while/lstm_cell_43/Sigmoid:y:0-qty_seq/while/lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� �
 qty_seq/while/lstm_cell_43/add_1AddV2"qty_seq/while/lstm_cell_43/mul:z:0$qty_seq/while/lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� �
$qty_seq/while/lstm_cell_43/Sigmoid_2Sigmoid)qty_seq/while/lstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� �
!qty_seq/while/lstm_cell_43/Relu_1Relu$qty_seq/while/lstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
 qty_seq/while/lstm_cell_43/mul_2Mul(qty_seq/while/lstm_cell_43/Sigmoid_2:y:0/qty_seq/while/lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
2qty_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemqty_seq_while_placeholder_1qty_seq_while_placeholder$qty_seq/while/lstm_cell_43/mul_2:z:0*
_output_shapes
: *
element_dtype0:���U
qty_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
qty_seq/while/addAddV2qty_seq_while_placeholderqty_seq/while/add/y:output:0*
T0*
_output_shapes
: W
qty_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
qty_seq/while/add_1AddV2(qty_seq_while_qty_seq_while_loop_counterqty_seq/while/add_1/y:output:0*
T0*
_output_shapes
: q
qty_seq/while/IdentityIdentityqty_seq/while/add_1:z:0^qty_seq/while/NoOp*
T0*
_output_shapes
: �
qty_seq/while/Identity_1Identity.qty_seq_while_qty_seq_while_maximum_iterations^qty_seq/while/NoOp*
T0*
_output_shapes
: q
qty_seq/while/Identity_2Identityqty_seq/while/add:z:0^qty_seq/while/NoOp*
T0*
_output_shapes
: �
qty_seq/while/Identity_3IdentityBqty_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^qty_seq/while/NoOp*
T0*
_output_shapes
: :����
qty_seq/while/Identity_4Identity$qty_seq/while/lstm_cell_43/mul_2:z:0^qty_seq/while/NoOp*
T0*'
_output_shapes
:��������� �
qty_seq/while/Identity_5Identity$qty_seq/while/lstm_cell_43/add_1:z:0^qty_seq/while/NoOp*
T0*'
_output_shapes
:��������� �
qty_seq/while/NoOpNoOp2^qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOp1^qty_seq/while/lstm_cell_43/MatMul/ReadVariableOp3^qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
qty_seq_while_identityqty_seq/while/Identity:output:0"=
qty_seq_while_identity_1!qty_seq/while/Identity_1:output:0"=
qty_seq_while_identity_2!qty_seq/while/Identity_2:output:0"=
qty_seq_while_identity_3!qty_seq/while/Identity_3:output:0"=
qty_seq_while_identity_4!qty_seq/while/Identity_4:output:0"=
qty_seq_while_identity_5!qty_seq/while/Identity_5:output:0"z
:qty_seq_while_lstm_cell_43_biasadd_readvariableop_resource<qty_seq_while_lstm_cell_43_biasadd_readvariableop_resource_0"|
;qty_seq_while_lstm_cell_43_matmul_1_readvariableop_resource=qty_seq_while_lstm_cell_43_matmul_1_readvariableop_resource_0"x
9qty_seq_while_lstm_cell_43_matmul_readvariableop_resource;qty_seq_while_lstm_cell_43_matmul_readvariableop_resource_0"P
%qty_seq_while_qty_seq_strided_slice_1'qty_seq_while_qty_seq_strided_slice_1_0"�
aqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensorcqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2f
1qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOp1qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOp2d
0qty_seq/while/lstm_cell_43/MatMul/ReadVariableOp0qty_seq/while/lstm_cell_43/MatMul/ReadVariableOp2h
2qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOp2qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�J
�
C__inference_qty_seq_layer_call_and_return_conditional_losses_488133

inputs>
+lstm_cell_43_matmul_readvariableop_resource:	�@
-lstm_cell_43_matmul_1_readvariableop_resource:	 �;
,lstm_cell_43_biasadd_readvariableop_resource:	�
identity��#lstm_cell_43/BiasAdd/ReadVariableOp�"lstm_cell_43/MatMul/ReadVariableOp�$lstm_cell_43/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_43/MatMul/ReadVariableOpReadVariableOp+lstm_cell_43_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_43/MatMulMatMulstrided_slice_2:output:0*lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_43_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
lstm_cell_43/MatMul_1MatMulzeros:output:0,lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_43/addAddV2lstm_cell_43/MatMul:product:0lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_43_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_43/BiasAddBiasAddlstm_cell_43/add:z:0+lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_43/splitSplit%lstm_cell_43/split/split_dim:output:0lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitn
lstm_cell_43/SigmoidSigmoidlstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� p
lstm_cell_43/Sigmoid_1Sigmoidlstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� w
lstm_cell_43/mulMullstm_cell_43/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� h
lstm_cell_43/ReluRelulstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
lstm_cell_43/mul_1Mullstm_cell_43/Sigmoid:y:0lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� {
lstm_cell_43/add_1AddV2lstm_cell_43/mul:z:0lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� p
lstm_cell_43/Sigmoid_2Sigmoidlstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� e
lstm_cell_43/Relu_1Relulstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
lstm_cell_43/mul_2Mullstm_cell_43/Sigmoid_2:y:0!lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_43_matmul_readvariableop_resource-lstm_cell_43_matmul_1_readvariableop_resource,lstm_cell_43_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_488049*
condR
while_cond_488048*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp$^lstm_cell_43/BiasAdd/ReadVariableOp#^lstm_cell_43/MatMul/ReadVariableOp%^lstm_cell_43/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_43/BiasAdd/ReadVariableOp#lstm_cell_43/BiasAdd/ReadVariableOp2H
"lstm_cell_43/MatMul/ReadVariableOp"lstm_cell_43/MatMul/ReadVariableOp2L
$lstm_cell_43/MatMul_1/ReadVariableOp$lstm_cell_43/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_485776

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
-__inference_lstm_cell_42_layer_call_fn_488379

inputs
states_0
states_1
unknown:	�
	unknown_0:	 �
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_42_layer_call_and_return_conditional_losses_484829o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:��������� q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:��������� :��������� : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/1
�
�
while_cond_487762
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_487762___redundant_placeholder04
0while_while_cond_487762___redundant_placeholder14
0while_while_cond_487762___redundant_placeholder24
0while_while_cond_487762___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
while_cond_485678
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_485678___redundant_placeholder04
0while_while_cond_485678___redundant_placeholder14
0while_while_cond_485678___redundant_placeholder24
0while_while_cond_485678___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
C__inference_model_6_layer_call_and_return_conditional_losses_485811

inputs
inputs_1!
qty_seq_485614:	�!
qty_seq_485616:	 �
qty_seq_485618:	�"
salt_seq_485764:	�"
salt_seq_485766:	 �
salt_seq_485768:	�"
salt_pred_485805:@
salt_pred_485807:
identity��qty_seq/StatefulPartitionedCall�!salt_pred/StatefulPartitionedCall� salt_seq/StatefulPartitionedCall�
qty_seq/StatefulPartitionedCallStatefulPartitionedCallinputs_1qty_seq_485614qty_seq_485616qty_seq_485618*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_485613�
 salt_seq/StatefulPartitionedCallStatefulPartitionedCallinputssalt_seq_485764salt_seq_485766salt_seq_485768*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_485763�
salt_seq_2/PartitionedCallPartitionedCall)salt_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_485776�
qty_seq_2/PartitionedCallPartitionedCall(qty_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_485783�
pattern/PartitionedCallPartitionedCall#salt_seq_2/PartitionedCall:output:0"qty_seq_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_pattern_layer_call_and_return_conditional_losses_485792�
!salt_pred/StatefulPartitionedCallStatefulPartitionedCall pattern/PartitionedCall:output:0salt_pred_485805salt_pred_485807*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_salt_pred_layer_call_and_return_conditional_losses_485804y
IdentityIdentity*salt_pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^qty_seq/StatefulPartitionedCall"^salt_pred/StatefulPartitionedCall!^salt_seq/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������:���������: : : : : : : : 2B
qty_seq/StatefulPartitionedCallqty_seq/StatefulPartitionedCall2F
!salt_pred/StatefulPartitionedCall!salt_pred/StatefulPartitionedCall2D
 salt_seq/StatefulPartitionedCall salt_seq/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_salt_seq_layer_call_fn_487077

inputs
unknown:	�
	unknown_0:	 �
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_485763o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
"model_6_salt_seq_while_cond_484667>
:model_6_salt_seq_while_model_6_salt_seq_while_loop_counterD
@model_6_salt_seq_while_model_6_salt_seq_while_maximum_iterations&
"model_6_salt_seq_while_placeholder(
$model_6_salt_seq_while_placeholder_1(
$model_6_salt_seq_while_placeholder_2(
$model_6_salt_seq_while_placeholder_3@
<model_6_salt_seq_while_less_model_6_salt_seq_strided_slice_1V
Rmodel_6_salt_seq_while_model_6_salt_seq_while_cond_484667___redundant_placeholder0V
Rmodel_6_salt_seq_while_model_6_salt_seq_while_cond_484667___redundant_placeholder1V
Rmodel_6_salt_seq_while_model_6_salt_seq_while_cond_484667___redundant_placeholder2V
Rmodel_6_salt_seq_while_model_6_salt_seq_while_cond_484667___redundant_placeholder3#
model_6_salt_seq_while_identity
�
model_6/salt_seq/while/LessLess"model_6_salt_seq_while_placeholder<model_6_salt_seq_while_less_model_6_salt_seq_strided_slice_1*
T0*
_output_shapes
: m
model_6/salt_seq/while/IdentityIdentitymodel_6/salt_seq/while/Less:z:0*
T0
*
_output_shapes
: "K
model_6_salt_seq_while_identity(model_6/salt_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�8
�
while_body_485965
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_42_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_42_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_42_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_42_matmul_readvariableop_resource:	�F
3while_lstm_cell_42_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_42_biasadd_readvariableop_resource:	���)while/lstm_cell_42/BiasAdd/ReadVariableOp�(while/lstm_cell_42/MatMul/ReadVariableOp�*while/lstm_cell_42/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_42/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_42/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_42_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
while/lstm_cell_42/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_42/addAddV2#while/lstm_cell_42/MatMul:product:0%while/lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_42_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_42/BiasAddBiasAddwhile/lstm_cell_42/add:z:01while/lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_42/splitSplit+while/lstm_cell_42/split/split_dim:output:0#while/lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitz
while/lstm_cell_42/SigmoidSigmoid!while/lstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_42/Sigmoid_1Sigmoid!while/lstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mulMul while/lstm_cell_42/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� t
while/lstm_cell_42/ReluRelu!while/lstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mul_1Mulwhile/lstm_cell_42/Sigmoid:y:0%while/lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/add_1AddV2while/lstm_cell_42/mul:z:0while/lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_42/Sigmoid_2Sigmoid!while/lstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� q
while/lstm_cell_42/Relu_1Reluwhile/lstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mul_2Mul while/lstm_cell_42/Sigmoid_2:y:0'while/lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_42/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���y
while/Identity_4Identitywhile/lstm_cell_42/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� y
while/Identity_5Identitywhile/lstm_cell_42/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp*^while/lstm_cell_42/BiasAdd/ReadVariableOp)^while/lstm_cell_42/MatMul/ReadVariableOp+^while/lstm_cell_42/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_42_biasadd_readvariableop_resource4while_lstm_cell_42_biasadd_readvariableop_resource_0"l
3while_lstm_cell_42_matmul_1_readvariableop_resource5while_lstm_cell_42_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_42_matmul_readvariableop_resource3while_lstm_cell_42_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_42/BiasAdd/ReadVariableOp)while/lstm_cell_42/BiasAdd/ReadVariableOp2T
(while/lstm_cell_42/MatMul/ReadVariableOp(while/lstm_cell_42/MatMul/ReadVariableOp2X
*while/lstm_cell_42/MatMul_1/ReadVariableOp*while/lstm_cell_42/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_lstm_cell_42_layer_call_fn_488396

inputs
states_0
states_1
unknown:	�
	unknown_0:	 �
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_42_layer_call_and_return_conditional_losses_484975o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:��������� q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:��������� :��������� : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/1
�B
�

salt_seq_while_body_486619.
*salt_seq_while_salt_seq_while_loop_counter4
0salt_seq_while_salt_seq_while_maximum_iterations
salt_seq_while_placeholder 
salt_seq_while_placeholder_1 
salt_seq_while_placeholder_2 
salt_seq_while_placeholder_3-
)salt_seq_while_salt_seq_strided_slice_1_0i
esalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0O
<salt_seq_while_lstm_cell_42_matmul_readvariableop_resource_0:	�Q
>salt_seq_while_lstm_cell_42_matmul_1_readvariableop_resource_0:	 �L
=salt_seq_while_lstm_cell_42_biasadd_readvariableop_resource_0:	�
salt_seq_while_identity
salt_seq_while_identity_1
salt_seq_while_identity_2
salt_seq_while_identity_3
salt_seq_while_identity_4
salt_seq_while_identity_5+
'salt_seq_while_salt_seq_strided_slice_1g
csalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensorM
:salt_seq_while_lstm_cell_42_matmul_readvariableop_resource:	�O
<salt_seq_while_lstm_cell_42_matmul_1_readvariableop_resource:	 �J
;salt_seq_while_lstm_cell_42_biasadd_readvariableop_resource:	���2salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOp�1salt_seq/while/lstm_cell_42/MatMul/ReadVariableOp�3salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOp�
@salt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2salt_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemesalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0salt_seq_while_placeholderIsalt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
1salt_seq/while/lstm_cell_42/MatMul/ReadVariableOpReadVariableOp<salt_seq_while_lstm_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
"salt_seq/while/lstm_cell_42/MatMulMatMul9salt_seq/while/TensorArrayV2Read/TensorListGetItem:item:09salt_seq/while/lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
3salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOp>salt_seq_while_lstm_cell_42_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
$salt_seq/while/lstm_cell_42/MatMul_1MatMulsalt_seq_while_placeholder_2;salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
salt_seq/while/lstm_cell_42/addAddV2,salt_seq/while/lstm_cell_42/MatMul:product:0.salt_seq/while/lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
2salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOp=salt_seq_while_lstm_cell_42_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
#salt_seq/while/lstm_cell_42/BiasAddBiasAdd#salt_seq/while/lstm_cell_42/add:z:0:salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
+salt_seq/while/lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
!salt_seq/while/lstm_cell_42/splitSplit4salt_seq/while/lstm_cell_42/split/split_dim:output:0,salt_seq/while/lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split�
#salt_seq/while/lstm_cell_42/SigmoidSigmoid*salt_seq/while/lstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� �
%salt_seq/while/lstm_cell_42/Sigmoid_1Sigmoid*salt_seq/while/lstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� �
salt_seq/while/lstm_cell_42/mulMul)salt_seq/while/lstm_cell_42/Sigmoid_1:y:0salt_seq_while_placeholder_3*
T0*'
_output_shapes
:��������� �
 salt_seq/while/lstm_cell_42/ReluRelu*salt_seq/while/lstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
!salt_seq/while/lstm_cell_42/mul_1Mul'salt_seq/while/lstm_cell_42/Sigmoid:y:0.salt_seq/while/lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� �
!salt_seq/while/lstm_cell_42/add_1AddV2#salt_seq/while/lstm_cell_42/mul:z:0%salt_seq/while/lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� �
%salt_seq/while/lstm_cell_42/Sigmoid_2Sigmoid*salt_seq/while/lstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� �
"salt_seq/while/lstm_cell_42/Relu_1Relu%salt_seq/while/lstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
!salt_seq/while/lstm_cell_42/mul_2Mul)salt_seq/while/lstm_cell_42/Sigmoid_2:y:00salt_seq/while/lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
3salt_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemsalt_seq_while_placeholder_1salt_seq_while_placeholder%salt_seq/while/lstm_cell_42/mul_2:z:0*
_output_shapes
: *
element_dtype0:���V
salt_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
salt_seq/while/addAddV2salt_seq_while_placeholdersalt_seq/while/add/y:output:0*
T0*
_output_shapes
: X
salt_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
salt_seq/while/add_1AddV2*salt_seq_while_salt_seq_while_loop_countersalt_seq/while/add_1/y:output:0*
T0*
_output_shapes
: t
salt_seq/while/IdentityIdentitysalt_seq/while/add_1:z:0^salt_seq/while/NoOp*
T0*
_output_shapes
: �
salt_seq/while/Identity_1Identity0salt_seq_while_salt_seq_while_maximum_iterations^salt_seq/while/NoOp*
T0*
_output_shapes
: t
salt_seq/while/Identity_2Identitysalt_seq/while/add:z:0^salt_seq/while/NoOp*
T0*
_output_shapes
: �
salt_seq/while/Identity_3IdentityCsalt_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^salt_seq/while/NoOp*
T0*
_output_shapes
: :����
salt_seq/while/Identity_4Identity%salt_seq/while/lstm_cell_42/mul_2:z:0^salt_seq/while/NoOp*
T0*'
_output_shapes
:��������� �
salt_seq/while/Identity_5Identity%salt_seq/while/lstm_cell_42/add_1:z:0^salt_seq/while/NoOp*
T0*'
_output_shapes
:��������� �
salt_seq/while/NoOpNoOp3^salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOp2^salt_seq/while/lstm_cell_42/MatMul/ReadVariableOp4^salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
salt_seq_while_identity salt_seq/while/Identity:output:0"?
salt_seq_while_identity_1"salt_seq/while/Identity_1:output:0"?
salt_seq_while_identity_2"salt_seq/while/Identity_2:output:0"?
salt_seq_while_identity_3"salt_seq/while/Identity_3:output:0"?
salt_seq_while_identity_4"salt_seq/while/Identity_4:output:0"?
salt_seq_while_identity_5"salt_seq/while/Identity_5:output:0"|
;salt_seq_while_lstm_cell_42_biasadd_readvariableop_resource=salt_seq_while_lstm_cell_42_biasadd_readvariableop_resource_0"~
<salt_seq_while_lstm_cell_42_matmul_1_readvariableop_resource>salt_seq_while_lstm_cell_42_matmul_1_readvariableop_resource_0"z
:salt_seq_while_lstm_cell_42_matmul_readvariableop_resource<salt_seq_while_lstm_cell_42_matmul_readvariableop_resource_0"T
'salt_seq_while_salt_seq_strided_slice_1)salt_seq_while_salt_seq_strided_slice_1_0"�
csalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensoresalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2h
2salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOp2salt_seq/while/lstm_cell_42/BiasAdd/ReadVariableOp2f
1salt_seq/while/lstm_cell_42/MatMul/ReadVariableOp1salt_seq/while/lstm_cell_42/MatMul/ReadVariableOp2j
3salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOp3salt_seq/while/lstm_cell_42/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�"
�
while_body_485384
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_43_485408_0:	�.
while_lstm_cell_43_485410_0:	 �*
while_lstm_cell_43_485412_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_43_485408:	�,
while_lstm_cell_43_485410:	 �(
while_lstm_cell_43_485412:	���*while/lstm_cell_43/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
*while/lstm_cell_43/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_43_485408_0while_lstm_cell_43_485410_0while_lstm_cell_43_485412_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_43_layer_call_and_return_conditional_losses_485325�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_43/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :����
while/Identity_4Identity3while/lstm_cell_43/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:��������� �
while/Identity_5Identity3while/lstm_cell_43/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:��������� y

while/NoOpNoOp+^while/lstm_cell_43/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_43_485408while_lstm_cell_43_485408_0"8
while_lstm_cell_43_485410while_lstm_cell_43_485410_0"8
while_lstm_cell_43_485412while_lstm_cell_43_485412_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2X
*while/lstm_cell_43/StatefulPartitionedCall*while/lstm_cell_43/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�8
�
while_body_487290
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_42_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_42_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_42_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_42_matmul_readvariableop_resource:	�F
3while_lstm_cell_42_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_42_biasadd_readvariableop_resource:	���)while/lstm_cell_42/BiasAdd/ReadVariableOp�(while/lstm_cell_42/MatMul/ReadVariableOp�*while/lstm_cell_42/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_42/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_42/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_42/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_42_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
while/lstm_cell_42/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_42/addAddV2#while/lstm_cell_42/MatMul:product:0%while/lstm_cell_42/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_42/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_42_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_42/BiasAddBiasAddwhile/lstm_cell_42/add:z:01while/lstm_cell_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_42/splitSplit+while/lstm_cell_42/split/split_dim:output:0#while/lstm_cell_42/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitz
while/lstm_cell_42/SigmoidSigmoid!while/lstm_cell_42/split:output:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_42/Sigmoid_1Sigmoid!while/lstm_cell_42/split:output:1*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mulMul while/lstm_cell_42/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� t
while/lstm_cell_42/ReluRelu!while/lstm_cell_42/split:output:2*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mul_1Mulwhile/lstm_cell_42/Sigmoid:y:0%while/lstm_cell_42/Relu:activations:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/add_1AddV2while/lstm_cell_42/mul:z:0while/lstm_cell_42/mul_1:z:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_42/Sigmoid_2Sigmoid!while/lstm_cell_42/split:output:3*
T0*'
_output_shapes
:��������� q
while/lstm_cell_42/Relu_1Reluwhile/lstm_cell_42/add_1:z:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_42/mul_2Mul while/lstm_cell_42/Sigmoid_2:y:0'while/lstm_cell_42/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_42/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���y
while/Identity_4Identitywhile/lstm_cell_42/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� y
while/Identity_5Identitywhile/lstm_cell_42/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp*^while/lstm_cell_42/BiasAdd/ReadVariableOp)^while/lstm_cell_42/MatMul/ReadVariableOp+^while/lstm_cell_42/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_42_biasadd_readvariableop_resource4while_lstm_cell_42_biasadd_readvariableop_resource_0"l
3while_lstm_cell_42_matmul_1_readvariableop_resource5while_lstm_cell_42_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_42_matmul_readvariableop_resource3while_lstm_cell_42_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_42/BiasAdd/ReadVariableOp)while/lstm_cell_42/BiasAdd/ReadVariableOp2T
(while/lstm_cell_42/MatMul/ReadVariableOp(while/lstm_cell_42/MatMul/ReadVariableOp2X
*while/lstm_cell_42/MatMul_1/ReadVariableOp*while/lstm_cell_42/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�F
�
__inference__traced_save_488675
file_prefix/
+savev2_salt_pred_kernel_read_readvariableop-
)savev2_salt_pred_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop;
7savev2_salt_seq_lstm_cell_42_kernel_read_readvariableopE
Asavev2_salt_seq_lstm_cell_42_recurrent_kernel_read_readvariableop9
5savev2_salt_seq_lstm_cell_42_bias_read_readvariableop:
6savev2_qty_seq_lstm_cell_43_kernel_read_readvariableopD
@savev2_qty_seq_lstm_cell_43_recurrent_kernel_read_readvariableop8
4savev2_qty_seq_lstm_cell_43_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_salt_pred_kernel_m_read_readvariableop4
0savev2_adam_salt_pred_bias_m_read_readvariableopB
>savev2_adam_salt_seq_lstm_cell_42_kernel_m_read_readvariableopL
Hsavev2_adam_salt_seq_lstm_cell_42_recurrent_kernel_m_read_readvariableop@
<savev2_adam_salt_seq_lstm_cell_42_bias_m_read_readvariableopA
=savev2_adam_qty_seq_lstm_cell_43_kernel_m_read_readvariableopK
Gsavev2_adam_qty_seq_lstm_cell_43_recurrent_kernel_m_read_readvariableop?
;savev2_adam_qty_seq_lstm_cell_43_bias_m_read_readvariableop6
2savev2_adam_salt_pred_kernel_v_read_readvariableop4
0savev2_adam_salt_pred_bias_v_read_readvariableopB
>savev2_adam_salt_seq_lstm_cell_42_kernel_v_read_readvariableopL
Hsavev2_adam_salt_seq_lstm_cell_42_recurrent_kernel_v_read_readvariableop@
<savev2_adam_salt_seq_lstm_cell_42_bias_v_read_readvariableopA
=savev2_adam_qty_seq_lstm_cell_43_kernel_v_read_readvariableopK
Gsavev2_adam_qty_seq_lstm_cell_43_recurrent_kernel_v_read_readvariableop?
;savev2_adam_qty_seq_lstm_cell_43_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_salt_pred_kernel_read_readvariableop)savev2_salt_pred_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop7savev2_salt_seq_lstm_cell_42_kernel_read_readvariableopAsavev2_salt_seq_lstm_cell_42_recurrent_kernel_read_readvariableop5savev2_salt_seq_lstm_cell_42_bias_read_readvariableop6savev2_qty_seq_lstm_cell_43_kernel_read_readvariableop@savev2_qty_seq_lstm_cell_43_recurrent_kernel_read_readvariableop4savev2_qty_seq_lstm_cell_43_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_salt_pred_kernel_m_read_readvariableop0savev2_adam_salt_pred_bias_m_read_readvariableop>savev2_adam_salt_seq_lstm_cell_42_kernel_m_read_readvariableopHsavev2_adam_salt_seq_lstm_cell_42_recurrent_kernel_m_read_readvariableop<savev2_adam_salt_seq_lstm_cell_42_bias_m_read_readvariableop=savev2_adam_qty_seq_lstm_cell_43_kernel_m_read_readvariableopGsavev2_adam_qty_seq_lstm_cell_43_recurrent_kernel_m_read_readvariableop;savev2_adam_qty_seq_lstm_cell_43_bias_m_read_readvariableop2savev2_adam_salt_pred_kernel_v_read_readvariableop0savev2_adam_salt_pred_bias_v_read_readvariableop>savev2_adam_salt_seq_lstm_cell_42_kernel_v_read_readvariableopHsavev2_adam_salt_seq_lstm_cell_42_recurrent_kernel_v_read_readvariableop<savev2_adam_salt_seq_lstm_cell_42_bias_v_read_readvariableop=savev2_adam_qty_seq_lstm_cell_43_kernel_v_read_readvariableopGsavev2_adam_qty_seq_lstm_cell_43_recurrent_kernel_v_read_readvariableop;savev2_adam_qty_seq_lstm_cell_43_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :@:: : : : : :	�:	 �:�:	�:	 �:�: : :@::	�:	 �:�:	�:	 �:�:@::	�:	 �:�:	�:	 �:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�:%	!

_output_shapes
:	 �:!


_output_shapes	
:�:%!

_output_shapes
:	�:%!

_output_shapes
:	 �:!

_output_shapes	
:�:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	 �:!

_output_shapes	
:�:%!

_output_shapes
:	�:%!

_output_shapes
:	 �:!

_output_shapes	
:�:$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	 �:!

_output_shapes	
:�:%!

_output_shapes
:	�:%!

_output_shapes
:	 �:!

_output_shapes	
:�: 

_output_shapes
: 
�
�
while_cond_488048
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_488048___redundant_placeholder04
0while_while_cond_488048___redundant_placeholder14
0while_while_cond_488048___redundant_placeholder24
0while_while_cond_488048___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
while_cond_485192
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_485192___redundant_placeholder04
0while_while_cond_485192___redundant_placeholder14
0while_while_cond_485192___redundant_placeholder24
0while_while_cond_485192___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�

e
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_485890

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:��������� Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

e
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_488303

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:��������� Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�J
�
C__inference_qty_seq_layer_call_and_return_conditional_losses_487990
inputs_0>
+lstm_cell_43_matmul_readvariableop_resource:	�@
-lstm_cell_43_matmul_1_readvariableop_resource:	 �;
,lstm_cell_43_biasadd_readvariableop_resource:	�
identity��#lstm_cell_43/BiasAdd/ReadVariableOp�"lstm_cell_43/MatMul/ReadVariableOp�$lstm_cell_43/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_43/MatMul/ReadVariableOpReadVariableOp+lstm_cell_43_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_43/MatMulMatMulstrided_slice_2:output:0*lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_43_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
lstm_cell_43/MatMul_1MatMulzeros:output:0,lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_43/addAddV2lstm_cell_43/MatMul:product:0lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_43_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_43/BiasAddBiasAddlstm_cell_43/add:z:0+lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_43/splitSplit%lstm_cell_43/split/split_dim:output:0lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitn
lstm_cell_43/SigmoidSigmoidlstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� p
lstm_cell_43/Sigmoid_1Sigmoidlstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� w
lstm_cell_43/mulMullstm_cell_43/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� h
lstm_cell_43/ReluRelulstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
lstm_cell_43/mul_1Mullstm_cell_43/Sigmoid:y:0lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� {
lstm_cell_43/add_1AddV2lstm_cell_43/mul:z:0lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� p
lstm_cell_43/Sigmoid_2Sigmoidlstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� e
lstm_cell_43/Relu_1Relulstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
lstm_cell_43/mul_2Mullstm_cell_43/Sigmoid_2:y:0!lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_43_matmul_readvariableop_resource-lstm_cell_43_matmul_1_readvariableop_resource,lstm_cell_43_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_487906*
condR
while_cond_487905*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp$^lstm_cell_43/BiasAdd/ReadVariableOp#^lstm_cell_43/MatMul/ReadVariableOp%^lstm_cell_43/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_43/BiasAdd/ReadVariableOp#lstm_cell_43/BiasAdd/ReadVariableOp2H
"lstm_cell_43/MatMul/ReadVariableOp"lstm_cell_43/MatMul/ReadVariableOp2L
$lstm_cell_43/MatMul_1/ReadVariableOp$lstm_cell_43/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�8
�
while_body_487763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_43_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_43_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_43_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_43_matmul_readvariableop_resource:	�F
3while_lstm_cell_43_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_43_biasadd_readvariableop_resource:	���)while/lstm_cell_43/BiasAdd/ReadVariableOp�(while/lstm_cell_43/MatMul/ReadVariableOp�*while/lstm_cell_43/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_43/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_43_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_43/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_43_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
while/lstm_cell_43/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_43/addAddV2#while/lstm_cell_43/MatMul:product:0%while/lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_43_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_43/BiasAddBiasAddwhile/lstm_cell_43/add:z:01while/lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_43/splitSplit+while/lstm_cell_43/split/split_dim:output:0#while/lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitz
while/lstm_cell_43/SigmoidSigmoid!while/lstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_43/Sigmoid_1Sigmoid!while/lstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mulMul while/lstm_cell_43/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� t
while/lstm_cell_43/ReluRelu!while/lstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mul_1Mulwhile/lstm_cell_43/Sigmoid:y:0%while/lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/add_1AddV2while/lstm_cell_43/mul:z:0while/lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_43/Sigmoid_2Sigmoid!while/lstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� q
while/lstm_cell_43/Relu_1Reluwhile/lstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mul_2Mul while/lstm_cell_43/Sigmoid_2:y:0'while/lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_43/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���y
while/Identity_4Identitywhile/lstm_cell_43/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� y
while/Identity_5Identitywhile/lstm_cell_43/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp*^while/lstm_cell_43/BiasAdd/ReadVariableOp)^while/lstm_cell_43/MatMul/ReadVariableOp+^while/lstm_cell_43/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_43_biasadd_readvariableop_resource4while_lstm_cell_43_biasadd_readvariableop_resource_0"l
3while_lstm_cell_43_matmul_1_readvariableop_resource5while_lstm_cell_43_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_43_matmul_readvariableop_resource3while_lstm_cell_43_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_43/BiasAdd/ReadVariableOp)while/lstm_cell_43/BiasAdd/ReadVariableOp2T
(while/lstm_cell_43/MatMul/ReadVariableOp(while/lstm_cell_43/MatMul/ReadVariableOp2X
*while/lstm_cell_43/MatMul_1/ReadVariableOp*while/lstm_cell_43/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�K
�
!model_6_qty_seq_while_body_484529<
8model_6_qty_seq_while_model_6_qty_seq_while_loop_counterB
>model_6_qty_seq_while_model_6_qty_seq_while_maximum_iterations%
!model_6_qty_seq_while_placeholder'
#model_6_qty_seq_while_placeholder_1'
#model_6_qty_seq_while_placeholder_2'
#model_6_qty_seq_while_placeholder_3;
7model_6_qty_seq_while_model_6_qty_seq_strided_slice_1_0w
smodel_6_qty_seq_while_tensorarrayv2read_tensorlistgetitem_model_6_qty_seq_tensorarrayunstack_tensorlistfromtensor_0V
Cmodel_6_qty_seq_while_lstm_cell_43_matmul_readvariableop_resource_0:	�X
Emodel_6_qty_seq_while_lstm_cell_43_matmul_1_readvariableop_resource_0:	 �S
Dmodel_6_qty_seq_while_lstm_cell_43_biasadd_readvariableop_resource_0:	�"
model_6_qty_seq_while_identity$
 model_6_qty_seq_while_identity_1$
 model_6_qty_seq_while_identity_2$
 model_6_qty_seq_while_identity_3$
 model_6_qty_seq_while_identity_4$
 model_6_qty_seq_while_identity_59
5model_6_qty_seq_while_model_6_qty_seq_strided_slice_1u
qmodel_6_qty_seq_while_tensorarrayv2read_tensorlistgetitem_model_6_qty_seq_tensorarrayunstack_tensorlistfromtensorT
Amodel_6_qty_seq_while_lstm_cell_43_matmul_readvariableop_resource:	�V
Cmodel_6_qty_seq_while_lstm_cell_43_matmul_1_readvariableop_resource:	 �Q
Bmodel_6_qty_seq_while_lstm_cell_43_biasadd_readvariableop_resource:	���9model_6/qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOp�8model_6/qty_seq/while/lstm_cell_43/MatMul/ReadVariableOp�:model_6/qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOp�
Gmodel_6/qty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
9model_6/qty_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsmodel_6_qty_seq_while_tensorarrayv2read_tensorlistgetitem_model_6_qty_seq_tensorarrayunstack_tensorlistfromtensor_0!model_6_qty_seq_while_placeholderPmodel_6/qty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
8model_6/qty_seq/while/lstm_cell_43/MatMul/ReadVariableOpReadVariableOpCmodel_6_qty_seq_while_lstm_cell_43_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
)model_6/qty_seq/while/lstm_cell_43/MatMulMatMul@model_6/qty_seq/while/TensorArrayV2Read/TensorListGetItem:item:0@model_6/qty_seq/while/lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:model_6/qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOpEmodel_6_qty_seq_while_lstm_cell_43_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
+model_6/qty_seq/while/lstm_cell_43/MatMul_1MatMul#model_6_qty_seq_while_placeholder_2Bmodel_6/qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&model_6/qty_seq/while/lstm_cell_43/addAddV23model_6/qty_seq/while/lstm_cell_43/MatMul:product:05model_6/qty_seq/while/lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
9model_6/qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOpDmodel_6_qty_seq_while_lstm_cell_43_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
*model_6/qty_seq/while/lstm_cell_43/BiasAddBiasAdd*model_6/qty_seq/while/lstm_cell_43/add:z:0Amodel_6/qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������t
2model_6/qty_seq/while/lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
(model_6/qty_seq/while/lstm_cell_43/splitSplit;model_6/qty_seq/while/lstm_cell_43/split/split_dim:output:03model_6/qty_seq/while/lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split�
*model_6/qty_seq/while/lstm_cell_43/SigmoidSigmoid1model_6/qty_seq/while/lstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� �
,model_6/qty_seq/while/lstm_cell_43/Sigmoid_1Sigmoid1model_6/qty_seq/while/lstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� �
&model_6/qty_seq/while/lstm_cell_43/mulMul0model_6/qty_seq/while/lstm_cell_43/Sigmoid_1:y:0#model_6_qty_seq_while_placeholder_3*
T0*'
_output_shapes
:��������� �
'model_6/qty_seq/while/lstm_cell_43/ReluRelu1model_6/qty_seq/while/lstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
(model_6/qty_seq/while/lstm_cell_43/mul_1Mul.model_6/qty_seq/while/lstm_cell_43/Sigmoid:y:05model_6/qty_seq/while/lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� �
(model_6/qty_seq/while/lstm_cell_43/add_1AddV2*model_6/qty_seq/while/lstm_cell_43/mul:z:0,model_6/qty_seq/while/lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� �
,model_6/qty_seq/while/lstm_cell_43/Sigmoid_2Sigmoid1model_6/qty_seq/while/lstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� �
)model_6/qty_seq/while/lstm_cell_43/Relu_1Relu,model_6/qty_seq/while/lstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
(model_6/qty_seq/while/lstm_cell_43/mul_2Mul0model_6/qty_seq/while/lstm_cell_43/Sigmoid_2:y:07model_6/qty_seq/while/lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
:model_6/qty_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#model_6_qty_seq_while_placeholder_1!model_6_qty_seq_while_placeholder,model_6/qty_seq/while/lstm_cell_43/mul_2:z:0*
_output_shapes
: *
element_dtype0:���]
model_6/qty_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
model_6/qty_seq/while/addAddV2!model_6_qty_seq_while_placeholder$model_6/qty_seq/while/add/y:output:0*
T0*
_output_shapes
: _
model_6/qty_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
model_6/qty_seq/while/add_1AddV28model_6_qty_seq_while_model_6_qty_seq_while_loop_counter&model_6/qty_seq/while/add_1/y:output:0*
T0*
_output_shapes
: �
model_6/qty_seq/while/IdentityIdentitymodel_6/qty_seq/while/add_1:z:0^model_6/qty_seq/while/NoOp*
T0*
_output_shapes
: �
 model_6/qty_seq/while/Identity_1Identity>model_6_qty_seq_while_model_6_qty_seq_while_maximum_iterations^model_6/qty_seq/while/NoOp*
T0*
_output_shapes
: �
 model_6/qty_seq/while/Identity_2Identitymodel_6/qty_seq/while/add:z:0^model_6/qty_seq/while/NoOp*
T0*
_output_shapes
: �
 model_6/qty_seq/while/Identity_3IdentityJmodel_6/qty_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model_6/qty_seq/while/NoOp*
T0*
_output_shapes
: :����
 model_6/qty_seq/while/Identity_4Identity,model_6/qty_seq/while/lstm_cell_43/mul_2:z:0^model_6/qty_seq/while/NoOp*
T0*'
_output_shapes
:��������� �
 model_6/qty_seq/while/Identity_5Identity,model_6/qty_seq/while/lstm_cell_43/add_1:z:0^model_6/qty_seq/while/NoOp*
T0*'
_output_shapes
:��������� �
model_6/qty_seq/while/NoOpNoOp:^model_6/qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOp9^model_6/qty_seq/while/lstm_cell_43/MatMul/ReadVariableOp;^model_6/qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "I
model_6_qty_seq_while_identity'model_6/qty_seq/while/Identity:output:0"M
 model_6_qty_seq_while_identity_1)model_6/qty_seq/while/Identity_1:output:0"M
 model_6_qty_seq_while_identity_2)model_6/qty_seq/while/Identity_2:output:0"M
 model_6_qty_seq_while_identity_3)model_6/qty_seq/while/Identity_3:output:0"M
 model_6_qty_seq_while_identity_4)model_6/qty_seq/while/Identity_4:output:0"M
 model_6_qty_seq_while_identity_5)model_6/qty_seq/while/Identity_5:output:0"�
Bmodel_6_qty_seq_while_lstm_cell_43_biasadd_readvariableop_resourceDmodel_6_qty_seq_while_lstm_cell_43_biasadd_readvariableop_resource_0"�
Cmodel_6_qty_seq_while_lstm_cell_43_matmul_1_readvariableop_resourceEmodel_6_qty_seq_while_lstm_cell_43_matmul_1_readvariableop_resource_0"�
Amodel_6_qty_seq_while_lstm_cell_43_matmul_readvariableop_resourceCmodel_6_qty_seq_while_lstm_cell_43_matmul_readvariableop_resource_0"p
5model_6_qty_seq_while_model_6_qty_seq_strided_slice_17model_6_qty_seq_while_model_6_qty_seq_strided_slice_1_0"�
qmodel_6_qty_seq_while_tensorarrayv2read_tensorlistgetitem_model_6_qty_seq_tensorarrayunstack_tensorlistfromtensorsmodel_6_qty_seq_while_tensorarrayv2read_tensorlistgetitem_model_6_qty_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2v
9model_6/qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOp9model_6/qty_seq/while/lstm_cell_43/BiasAdd/ReadVariableOp2t
8model_6/qty_seq/while/lstm_cell_43/MatMul/ReadVariableOp8model_6/qty_seq/while/lstm_cell_43/MatMul/ReadVariableOp2x
:model_6/qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOp:model_6/qty_seq/while/lstm_cell_43/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�8
�
while_body_487906
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_43_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_43_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_43_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_43_matmul_readvariableop_resource:	�F
3while_lstm_cell_43_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_43_biasadd_readvariableop_resource:	���)while/lstm_cell_43/BiasAdd/ReadVariableOp�(while/lstm_cell_43/MatMul/ReadVariableOp�*while/lstm_cell_43/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_43/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_43_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_43/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_43/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_43_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype0�
while/lstm_cell_43/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_43/addAddV2#while/lstm_cell_43/MatMul:product:0%while/lstm_cell_43/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_43/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_43_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_43/BiasAddBiasAddwhile/lstm_cell_43/add:z:01while/lstm_cell_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_43/splitSplit+while/lstm_cell_43/split/split_dim:output:0#while/lstm_cell_43/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_splitz
while/lstm_cell_43/SigmoidSigmoid!while/lstm_cell_43/split:output:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_43/Sigmoid_1Sigmoid!while/lstm_cell_43/split:output:1*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mulMul while/lstm_cell_43/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� t
while/lstm_cell_43/ReluRelu!while/lstm_cell_43/split:output:2*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mul_1Mulwhile/lstm_cell_43/Sigmoid:y:0%while/lstm_cell_43/Relu:activations:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/add_1AddV2while/lstm_cell_43/mul:z:0while/lstm_cell_43/mul_1:z:0*
T0*'
_output_shapes
:��������� |
while/lstm_cell_43/Sigmoid_2Sigmoid!while/lstm_cell_43/split:output:3*
T0*'
_output_shapes
:��������� q
while/lstm_cell_43/Relu_1Reluwhile/lstm_cell_43/add_1:z:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell_43/mul_2Mul while/lstm_cell_43/Sigmoid_2:y:0'while/lstm_cell_43/Relu_1:activations:0*
T0*'
_output_shapes
:��������� �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_43/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���y
while/Identity_4Identitywhile/lstm_cell_43/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� y
while/Identity_5Identitywhile/lstm_cell_43/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� �

while/NoOpNoOp*^while/lstm_cell_43/BiasAdd/ReadVariableOp)^while/lstm_cell_43/MatMul/ReadVariableOp+^while/lstm_cell_43/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_43_biasadd_readvariableop_resource4while_lstm_cell_43_biasadd_readvariableop_resource_0"l
3while_lstm_cell_43_matmul_1_readvariableop_resource5while_lstm_cell_43_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_43_matmul_readvariableop_resource3while_lstm_cell_43_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_43/BiasAdd/ReadVariableOp)while/lstm_cell_43/BiasAdd/ReadVariableOp2T
(while/lstm_cell_43/MatMul/ReadVariableOp(while/lstm_cell_43/MatMul/ReadVariableOp2X
*while/lstm_cell_43/MatMul_1/ReadVariableOp*while/lstm_cell_43/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_486129
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_486129___redundant_placeholder04
0while_while_cond_486129___redundant_placeholder14
0while_while_cond_486129___redundant_placeholder24
0while_while_cond_486129___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
while_cond_488191
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_488191___redundant_placeholder04
0while_while_cond_488191___redundant_placeholder14
0while_while_cond_488191___redundant_placeholder24
0while_while_cond_488191___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
d
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_488291

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
K
quantity_data:
serving_default_quantity_data:0���������
C
	salt_data6
serving_default_salt_data:0���������=
	salt_pred0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
�
cell

state_spec
	variables
trainable_variables
regularization_losses
 	keras_api
!_random_generator
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(_random_generator
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/_random_generator
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
�

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
�
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_rate8m�9m�Em�Fm�Gm�Hm�Im�Jm�8v�9v�Ev�Fv�Gv�Hv�Iv�Jv�"
	optimizer
X
E0
F1
G2
H3
I4
J5
86
97"
trackable_list_wrapper
X
E0
F1
G2
H3
I4
J5
86
97"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_model_6_layer_call_fn_485830
(__inference_model_6_layer_call_fn_486398
(__inference_model_6_layer_call_fn_486420
(__inference_model_6_layer_call_fn_486316�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_model_6_layer_call_and_return_conditional_losses_486713
C__inference_model_6_layer_call_and_return_conditional_losses_487020
C__inference_model_6_layer_call_and_return_conditional_losses_486343
C__inference_model_6_layer_call_and_return_conditional_losses_486370�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
!__inference__wrapped_model_484762	salt_dataquantity_data"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
Pserving_default"
signature_map
�
Q
state_size

Ekernel
Frecurrent_kernel
Gbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V_random_generator
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
E0
F1
G2"
trackable_list_wrapper
5
E0
F1
G2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Ystates
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
)__inference_salt_seq_layer_call_fn_487055
)__inference_salt_seq_layer_call_fn_487066
)__inference_salt_seq_layer_call_fn_487077
)__inference_salt_seq_layer_call_fn_487088�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_salt_seq_layer_call_and_return_conditional_losses_487231
D__inference_salt_seq_layer_call_and_return_conditional_losses_487374
D__inference_salt_seq_layer_call_and_return_conditional_losses_487517
D__inference_salt_seq_layer_call_and_return_conditional_losses_487660�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�
_
state_size

Hkernel
Irecurrent_kernel
Jbias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d_random_generator
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

gstates
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
(__inference_qty_seq_layer_call_fn_487671
(__inference_qty_seq_layer_call_fn_487682
(__inference_qty_seq_layer_call_fn_487693
(__inference_qty_seq_layer_call_fn_487704�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_qty_seq_layer_call_and_return_conditional_losses_487847
C__inference_qty_seq_layer_call_and_return_conditional_losses_487990
C__inference_qty_seq_layer_call_and_return_conditional_losses_488133
C__inference_qty_seq_layer_call_and_return_conditional_losses_488276�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
$	variables
%trainable_variables
&regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
+__inference_salt_seq_2_layer_call_fn_488281
+__inference_salt_seq_2_layer_call_fn_488286�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_488291
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_488303�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
+	variables
,trainable_variables
-regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
*__inference_qty_seq_2_layer_call_fn_488308
*__inference_qty_seq_2_layer_call_fn_488313�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_488318
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_488330�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_pattern_layer_call_fn_488336�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_pattern_layer_call_and_return_conditional_losses_488343�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
": @2salt_pred/kernel
:2salt_pred/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_salt_pred_layer_call_fn_488352�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_salt_pred_layer_call_and_return_conditional_losses_488362�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
/:-	�2salt_seq/lstm_cell_42/kernel
9:7	 �2&salt_seq/lstm_cell_42/recurrent_kernel
):'�2salt_seq/lstm_cell_42/bias
.:,	�2qty_seq/lstm_cell_43/kernel
8:6	 �2%qty_seq/lstm_cell_43/recurrent_kernel
(:&�2qty_seq/lstm_cell_43/bias
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_signature_wrapper_487044quantity_data	salt_data"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
5
E0
F1
G2"
trackable_list_wrapper
5
E0
F1
G2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
R	variables
Strainable_variables
Tregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
-__inference_lstm_cell_42_layer_call_fn_488379
-__inference_lstm_cell_42_layer_call_fn_488396�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_lstm_cell_42_layer_call_and_return_conditional_losses_488428
H__inference_lstm_cell_42_layer_call_and_return_conditional_losses_488460�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
-__inference_lstm_cell_43_layer_call_fn_488477
-__inference_lstm_cell_43_layer_call_fn_488494�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_lstm_cell_43_layer_call_and_return_conditional_losses_488526
H__inference_lstm_cell_43_layer_call_and_return_conditional_losses_488558�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
':%@2Adam/salt_pred/kernel/m
!:2Adam/salt_pred/bias/m
4:2	�2#Adam/salt_seq/lstm_cell_42/kernel/m
>:<	 �2-Adam/salt_seq/lstm_cell_42/recurrent_kernel/m
.:,�2!Adam/salt_seq/lstm_cell_42/bias/m
3:1	�2"Adam/qty_seq/lstm_cell_43/kernel/m
=:;	 �2,Adam/qty_seq/lstm_cell_43/recurrent_kernel/m
-:+�2 Adam/qty_seq/lstm_cell_43/bias/m
':%@2Adam/salt_pred/kernel/v
!:2Adam/salt_pred/bias/v
4:2	�2#Adam/salt_seq/lstm_cell_42/kernel/v
>:<	 �2-Adam/salt_seq/lstm_cell_42/recurrent_kernel/v
.:,�2!Adam/salt_seq/lstm_cell_42/bias/v
3:1	�2"Adam/qty_seq/lstm_cell_43/kernel/v
=:;	 �2,Adam/qty_seq/lstm_cell_43/recurrent_kernel/v
-:+�2 Adam/qty_seq/lstm_cell_43/bias/v�
!__inference__wrapped_model_484762�HIJEFG89h�e
^�[
Y�V
'�$
	salt_data���������
+�(
quantity_data���������
� "5�2
0
	salt_pred#� 
	salt_pred����������
H__inference_lstm_cell_42_layer_call_and_return_conditional_losses_488428�EFG��}
v�s
 �
inputs���������
K�H
"�
states/0��������� 
"�
states/1��������� 
p 
� "s�p
i�f
�
0/0��������� 
E�B
�
0/1/0��������� 
�
0/1/1��������� 
� �
H__inference_lstm_cell_42_layer_call_and_return_conditional_losses_488460�EFG��}
v�s
 �
inputs���������
K�H
"�
states/0��������� 
"�
states/1��������� 
p
� "s�p
i�f
�
0/0��������� 
E�B
�
0/1/0��������� 
�
0/1/1��������� 
� �
-__inference_lstm_cell_42_layer_call_fn_488379�EFG��}
v�s
 �
inputs���������
K�H
"�
states/0��������� 
"�
states/1��������� 
p 
� "c�`
�
0��������� 
A�>
�
1/0��������� 
�
1/1��������� �
-__inference_lstm_cell_42_layer_call_fn_488396�EFG��}
v�s
 �
inputs���������
K�H
"�
states/0��������� 
"�
states/1��������� 
p
� "c�`
�
0��������� 
A�>
�
1/0��������� 
�
1/1��������� �
H__inference_lstm_cell_43_layer_call_and_return_conditional_losses_488526�HIJ��}
v�s
 �
inputs���������
K�H
"�
states/0��������� 
"�
states/1��������� 
p 
� "s�p
i�f
�
0/0��������� 
E�B
�
0/1/0��������� 
�
0/1/1��������� 
� �
H__inference_lstm_cell_43_layer_call_and_return_conditional_losses_488558�HIJ��}
v�s
 �
inputs���������
K�H
"�
states/0��������� 
"�
states/1��������� 
p
� "s�p
i�f
�
0/0��������� 
E�B
�
0/1/0��������� 
�
0/1/1��������� 
� �
-__inference_lstm_cell_43_layer_call_fn_488477�HIJ��}
v�s
 �
inputs���������
K�H
"�
states/0��������� 
"�
states/1��������� 
p 
� "c�`
�
0��������� 
A�>
�
1/0��������� 
�
1/1��������� �
-__inference_lstm_cell_43_layer_call_fn_488494�HIJ��}
v�s
 �
inputs���������
K�H
"�
states/0��������� 
"�
states/1��������� 
p
� "c�`
�
0��������� 
A�>
�
1/0��������� 
�
1/1��������� �
C__inference_model_6_layer_call_and_return_conditional_losses_486343�HIJEFG89p�m
f�c
Y�V
'�$
	salt_data���������
+�(
quantity_data���������
p 

 
� "%�"
�
0���������
� �
C__inference_model_6_layer_call_and_return_conditional_losses_486370�HIJEFG89p�m
f�c
Y�V
'�$
	salt_data���������
+�(
quantity_data���������
p

 
� "%�"
�
0���������
� �
C__inference_model_6_layer_call_and_return_conditional_losses_486713�HIJEFG89j�g
`�]
S�P
&�#
inputs/0���������
&�#
inputs/1���������
p 

 
� "%�"
�
0���������
� �
C__inference_model_6_layer_call_and_return_conditional_losses_487020�HIJEFG89j�g
`�]
S�P
&�#
inputs/0���������
&�#
inputs/1���������
p

 
� "%�"
�
0���������
� �
(__inference_model_6_layer_call_fn_485830�HIJEFG89p�m
f�c
Y�V
'�$
	salt_data���������
+�(
quantity_data���������
p 

 
� "�����������
(__inference_model_6_layer_call_fn_486316�HIJEFG89p�m
f�c
Y�V
'�$
	salt_data���������
+�(
quantity_data���������
p

 
� "�����������
(__inference_model_6_layer_call_fn_486398�HIJEFG89j�g
`�]
S�P
&�#
inputs/0���������
&�#
inputs/1���������
p 

 
� "�����������
(__inference_model_6_layer_call_fn_486420�HIJEFG89j�g
`�]
S�P
&�#
inputs/0���������
&�#
inputs/1���������
p

 
� "�����������
C__inference_pattern_layer_call_and_return_conditional_losses_488343�Z�W
P�M
K�H
"�
inputs/0��������� 
"�
inputs/1��������� 
� "%�"
�
0���������@
� �
(__inference_pattern_layer_call_fn_488336vZ�W
P�M
K�H
"�
inputs/0��������� 
"�
inputs/1��������� 
� "����������@�
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_488318\3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_488330\3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� }
*__inference_qty_seq_2_layer_call_fn_488308O3�0
)�&
 �
inputs��������� 
p 
� "���������� }
*__inference_qty_seq_2_layer_call_fn_488313O3�0
)�&
 �
inputs��������� 
p
� "���������� �
C__inference_qty_seq_layer_call_and_return_conditional_losses_487847}HIJO�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "%�"
�
0��������� 
� �
C__inference_qty_seq_layer_call_and_return_conditional_losses_487990}HIJO�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "%�"
�
0��������� 
� �
C__inference_qty_seq_layer_call_and_return_conditional_losses_488133mHIJ?�<
5�2
$�!
inputs���������

 
p 

 
� "%�"
�
0��������� 
� �
C__inference_qty_seq_layer_call_and_return_conditional_losses_488276mHIJ?�<
5�2
$�!
inputs���������

 
p

 
� "%�"
�
0��������� 
� �
(__inference_qty_seq_layer_call_fn_487671pHIJO�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "���������� �
(__inference_qty_seq_layer_call_fn_487682pHIJO�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "���������� �
(__inference_qty_seq_layer_call_fn_487693`HIJ?�<
5�2
$�!
inputs���������

 
p 

 
� "���������� �
(__inference_qty_seq_layer_call_fn_487704`HIJ?�<
5�2
$�!
inputs���������

 
p

 
� "���������� �
E__inference_salt_pred_layer_call_and_return_conditional_losses_488362\89/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� }
*__inference_salt_pred_layer_call_fn_488352O89/�,
%�"
 �
inputs���������@
� "�����������
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_488291\3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_488303\3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� ~
+__inference_salt_seq_2_layer_call_fn_488281O3�0
)�&
 �
inputs��������� 
p 
� "���������� ~
+__inference_salt_seq_2_layer_call_fn_488286O3�0
)�&
 �
inputs��������� 
p
� "���������� �
D__inference_salt_seq_layer_call_and_return_conditional_losses_487231}EFGO�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "%�"
�
0��������� 
� �
D__inference_salt_seq_layer_call_and_return_conditional_losses_487374}EFGO�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "%�"
�
0��������� 
� �
D__inference_salt_seq_layer_call_and_return_conditional_losses_487517mEFG?�<
5�2
$�!
inputs���������

 
p 

 
� "%�"
�
0��������� 
� �
D__inference_salt_seq_layer_call_and_return_conditional_losses_487660mEFG?�<
5�2
$�!
inputs���������

 
p

 
� "%�"
�
0��������� 
� �
)__inference_salt_seq_layer_call_fn_487055pEFGO�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "���������� �
)__inference_salt_seq_layer_call_fn_487066pEFGO�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "���������� �
)__inference_salt_seq_layer_call_fn_487077`EFG?�<
5�2
$�!
inputs���������

 
p 

 
� "���������� �
)__inference_salt_seq_layer_call_fn_487088`EFG?�<
5�2
$�!
inputs���������

 
p

 
� "���������� �
$__inference_signature_wrapper_487044�HIJEFG89��~
� 
w�t
<
quantity_data+�(
quantity_data���������
4
	salt_data'�$
	salt_data���������"5�2
0
	salt_pred#� 
	salt_pred���������