��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8��

y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	�@*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:@*
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:@ *
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
: *
dtype0
y
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	 �*
dtype0
q
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_9/bias
j
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes	
:�*
dtype0
|
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_10/kernel
u
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel* 
_output_shapes
:
��*
dtype0
s
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_10/bias
l
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes	
:�*
dtype0
|
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_11/kernel
u
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel* 
_output_shapes
:
��*
dtype0
s
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_11/bias
l
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes	
:�*
dtype0
|
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_12/kernel
u
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel* 
_output_shapes
:
��*
dtype0
s
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_12/bias
l
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes	
:�*
dtype0
{
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� * 
shared_namedense_13/kernel
t
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes
:	� *
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
: *
dtype0
�
class_output_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameclass_output_1/kernel

)class_output_1/kernel/Read/ReadVariableOpReadVariableOpclass_output_1/kernel*
_output_shapes

: *
dtype0
~
class_output_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameclass_output_1/bias
w
'class_output_1/bias/Read/ReadVariableOpReadVariableOpclass_output_1/bias*
_output_shapes
:*
dtype0
~
residual_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_nameresidual_1/kernel
w
%residual_1/kernel/Read/ReadVariableOpReadVariableOpresidual_1/kernel*
_output_shapes

: *
dtype0
v
residual_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameresidual_1/bias
o
#residual_1/bias/Read/ReadVariableOpReadVariableOpresidual_1/bias*
_output_shapes
:*
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
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*&
shared_nameAdam/dense_7/kernel/m
�
)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes
:	�@*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/dense_8/kernel/m

)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes

:@ *
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
: *
dtype0
�
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*&
shared_nameAdam/dense_9/kernel/m
�
)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes
:	 �*
dtype0

Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/dense_9/bias/m
x
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_10/kernel/m
�
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_10/bias/m
z
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_11/kernel/m
�
*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_11/bias/m
z
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_12/kernel/m
�
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_12/bias/m
z
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *'
shared_nameAdam/dense_13/kernel/m
�
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes
:	� *
dtype0
�
Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
: *
dtype0
�
Adam/class_output_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_nameAdam/class_output_1/kernel/m
�
0Adam/class_output_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/class_output_1/kernel/m*
_output_shapes

: *
dtype0
�
Adam/class_output_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/class_output_1/bias/m
�
.Adam/class_output_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/class_output_1/bias/m*
_output_shapes
:*
dtype0
�
Adam/residual_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_nameAdam/residual_1/kernel/m
�
,Adam/residual_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/residual_1/kernel/m*
_output_shapes

: *
dtype0
�
Adam/residual_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/residual_1/bias/m
}
*Adam/residual_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/residual_1/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*&
shared_nameAdam/dense_7/kernel/v
�
)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes
:	�@*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:@*
dtype0
�
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/dense_8/kernel/v

)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes

:@ *
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
: *
dtype0
�
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*&
shared_nameAdam/dense_9/kernel/v
�
)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes
:	 �*
dtype0

Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/dense_9/bias/v
x
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_10/kernel/v
�
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_10/bias/v
z
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_11/kernel/v
�
*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_11/bias/v
z
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/dense_12/kernel/v
�
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_12/bias/v
z
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *'
shared_nameAdam/dense_13/kernel/v
�
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes
:	� *
dtype0
�
Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
: *
dtype0
�
Adam/class_output_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_nameAdam/class_output_1/kernel/v
�
0Adam/class_output_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/class_output_1/kernel/v*
_output_shapes

: *
dtype0
�
Adam/class_output_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/class_output_1/bias/v
�
.Adam/class_output_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/class_output_1/bias/v*
_output_shapes
:*
dtype0
�
Adam/residual_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_nameAdam/residual_1/kernel/v
�
,Adam/residual_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/residual_1/kernel/v*
_output_shapes

: *
dtype0
�
Adam/residual_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/residual_1/bias/v
}
*Adam/residual_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/residual_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�Z
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�Z
value�ZB�Z B�Z
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer-10
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
h

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
h

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
h

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
h

6kernel
7bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
h

<kernel
=bias
>trainable_variables
?regularization_losses
@	variables
A	keras_api
h

Bkernel
Cbias
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
R
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
�
Liter

Mbeta_1

Nbeta_2
	Odecay
Plearning_ratem�m�m�m�m�m�$m�%m�*m�+m�0m�1m�6m�7m�<m�=m�Bm�Cm�v�v�v�v�v�v�$v�%v�*v�+v�0v�1v�6v�7v�<v�=v�Bv�Cv�
�
0
1
2
3
4
5
$6
%7
*8
+9
010
111
612
713
<14
=15
B16
C17
 
�
0
1
2
3
4
5
$6
%7
*8
+9
010
111
612
713
<14
=15
B16
C17
�
trainable_variables
Qmetrics
Rnon_trainable_variables

Slayers
regularization_losses
Tlayer_regularization_losses
	variables
Ulayer_metrics
 
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
trainable_variables
Vmetrics
Wnon_trainable_variables

Xlayers
regularization_losses
Ylayer_regularization_losses
	variables
Zlayer_metrics
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
trainable_variables
[metrics
\non_trainable_variables

]layers
regularization_losses
^layer_regularization_losses
	variables
_layer_metrics
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
 trainable_variables
`metrics
anon_trainable_variables

blayers
!regularization_losses
clayer_regularization_losses
"	variables
dlayer_metrics
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
�
&trainable_variables
emetrics
fnon_trainable_variables

glayers
'regularization_losses
hlayer_regularization_losses
(	variables
ilayer_metrics
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1
 

*0
+1
�
,trainable_variables
jmetrics
knon_trainable_variables

llayers
-regularization_losses
mlayer_regularization_losses
.	variables
nlayer_metrics
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
�
2trainable_variables
ometrics
pnon_trainable_variables

qlayers
3regularization_losses
rlayer_regularization_losses
4	variables
slayer_metrics
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71
 

60
71
�
8trainable_variables
tmetrics
unon_trainable_variables

vlayers
9regularization_losses
wlayer_regularization_losses
:	variables
xlayer_metrics
a_
VARIABLE_VALUEclass_output_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEclass_output_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

<0
=1
 

<0
=1
�
>trainable_variables
ymetrics
znon_trainable_variables

{layers
?regularization_losses
|layer_regularization_losses
@	variables
}layer_metrics
][
VARIABLE_VALUEresidual_1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEresidual_1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
 

B0
C1
�
Dtrainable_variables
~metrics
non_trainable_variables
�layers
Eregularization_losses
 �layer_regularization_losses
F	variables
�layer_metrics
 
 
 
�
Htrainable_variables
�metrics
�non_trainable_variables
�layers
Iregularization_losses
 �layer_regularization_losses
J	variables
�layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

�0
 
N
0
1
2
3
4
5
6
7
	8

9
10
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
}{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/class_output_1/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/class_output_1/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/residual_1/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/residual_1/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/class_output_1/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/class_output_1/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/residual_1/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/residual_1/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_2Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2dense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasclass_output_1/kernelclass_output_1/biasresidual_1/kernelresidual_1/bias*
Tin
2*
Tout
2*'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_699784
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp)class_output_1/kernel/Read/ReadVariableOp'class_output_1/bias/Read/ReadVariableOp%residual_1/kernel/Read/ReadVariableOp#residual_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp0Adam/class_output_1/kernel/m/Read/ReadVariableOp.Adam/class_output_1/bias/m/Read/ReadVariableOp,Adam/residual_1/kernel/m/Read/ReadVariableOp*Adam/residual_1/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOp0Adam/class_output_1/kernel/v/Read/ReadVariableOp.Adam/class_output_1/bias/v/Read/ReadVariableOp,Adam/residual_1/kernel/v/Read/ReadVariableOp*Adam/residual_1/bias/v/Read/ReadVariableOpConst*J
TinC
A2?	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__traced_save_700404
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasclass_output_1/kernelclass_output_1/biasresidual_1/kernelresidual_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/dense_8/kernel/mAdam/dense_8/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/mAdam/dense_10/kernel/mAdam/dense_10/bias/mAdam/dense_11/kernel/mAdam/dense_11/bias/mAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/mAdam/class_output_1/kernel/mAdam/class_output_1/bias/mAdam/residual_1/kernel/mAdam/residual_1/bias/mAdam/dense_7/kernel/vAdam/dense_7/bias/vAdam/dense_8/kernel/vAdam/dense_8/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/vAdam/dense_10/kernel/vAdam/dense_10/bias/vAdam/dense_11/kernel/vAdam/dense_11/bias/vAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/vAdam/class_output_1/kernel/vAdam/class_output_1/bias/vAdam/residual_1/kernel/vAdam/residual_1/bias/v*I
TinB
@2>*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__traced_restore_700599��
�
�
D__inference_dense_13_layer_call_and_return_conditional_losses_700133

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_model_1_layer_call_fn_699733
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_6996942
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:����������::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
C__inference_dense_9_layer_call_and_return_conditional_losses_700053

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 �*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :::O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�4
�
C__inference_model_1_layer_call_and_return_conditional_losses_699500
input_2
dense_7_699263
dense_7_699265
dense_8_699290
dense_8_699292
dense_9_699317
dense_9_699319
dense_10_699344
dense_10_699346
dense_11_699371
dense_11_699373
dense_12_699398
dense_12_699400
dense_13_699425
dense_13_699427
class_output_699452
class_output_699454
residual_699478
residual_699480
identity��$class_output/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall� residual/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_7_699263dense_7_699265*
Tin
2*
Tout
2*'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_6992522!
dense_7/StatefulPartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_699290dense_8_699292*
Tin
2*
Tout
2*'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_6992792!
dense_8/StatefulPartitionedCall�
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_699317dense_9_699319*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_6993062!
dense_9/StatefulPartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_699344dense_10_699346*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_6993332"
 dense_10/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_699371dense_11_699373*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_6993602"
 dense_11/StatefulPartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_699398dense_12_699400*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_6993872"
 dense_12/StatefulPartitionedCall�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_699425dense_13_699427*
Tin
2*
Tout
2*'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_6994142"
 dense_13/StatefulPartitionedCall�
$class_output/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0class_output_699452class_output_699454*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_class_output_layer_call_and_return_conditional_losses_6994412&
$class_output/StatefulPartitionedCall�
 residual/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0residual_699478residual_699480*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_residual_layer_call_and_return_conditional_losses_6994672"
 residual/StatefulPartitionedCall�
concatenate_1/PartitionedCallPartitionedCall-class_output/StatefulPartitionedCall:output:0)residual/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_6994902
concatenate_1/PartitionedCall�
IdentityIdentity&concatenate_1/PartitionedCall:output:0%^class_output/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall!^residual/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:����������::::::::::::::::::2L
$class_output/StatefulPartitionedCall$class_output/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2D
 residual/StatefulPartitionedCall residual/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
~
)__inference_dense_12_layer_call_fn_700122

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_6993872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_model_1_layer_call_fn_699961

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_6996032
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:����������::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_class_output_layer_call_and_return_conditional_losses_699441

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :::O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_model_1_layer_call_fn_700002

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_6996942
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:����������::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
D__inference_dense_11_layer_call_and_return_conditional_losses_699360

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
~
)__inference_residual_layer_call_fn_700181

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_residual_layer_call_and_return_conditional_losses_6994672
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
D__inference_dense_12_layer_call_and_return_conditional_losses_699387

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
D__inference_residual_layer_call_and_return_conditional_losses_699467

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :::O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
u
I__inference_concatenate_1_layer_call_and_return_conditional_losses_700188
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
$__inference_signature_wrapper_699784
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__wrapped_model_6992372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:����������::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
~
)__inference_dense_11_layer_call_fn_700102

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_6993602
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_class_output_layer_call_fn_700162

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_class_output_layer_call_and_return_conditional_losses_6994412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
D__inference_dense_12_layer_call_and_return_conditional_losses_700113

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
D__inference_dense_13_layer_call_and_return_conditional_losses_699414

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
s
I__inference_concatenate_1_layer_call_and_return_conditional_losses_699490

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
__inference__traced_save_700404
file_prefix-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop4
0savev2_class_output_1_kernel_read_readvariableop2
.savev2_class_output_1_bias_read_readvariableop0
,savev2_residual_1_kernel_read_readvariableop.
*savev2_residual_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop3
/savev2_adam_dense_11_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop;
7savev2_adam_class_output_1_kernel_m_read_readvariableop9
5savev2_adam_class_output_1_bias_m_read_readvariableop7
3savev2_adam_residual_1_kernel_m_read_readvariableop5
1savev2_adam_residual_1_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableop5
1savev2_adam_dense_11_kernel_v_read_readvariableop3
/savev2_adam_dense_11_bias_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop;
7savev2_adam_class_output_1_kernel_v_read_readvariableop9
5savev2_adam_class_output_1_bias_v_read_readvariableop7
3savev2_adam_residual_1_kernel_v_read_readvariableop5
1savev2_adam_residual_1_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_9168b7fa4b4745d5b23686fb744a6402/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�!
value�!B�!=B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop0savev2_class_output_1_kernel_read_readvariableop.savev2_class_output_1_bias_read_readvariableop,savev2_residual_1_kernel_read_readvariableop*savev2_residual_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop/savev2_adam_dense_11_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop7savev2_adam_class_output_1_kernel_m_read_readvariableop5savev2_adam_class_output_1_bias_m_read_readvariableop3savev2_adam_residual_1_kernel_m_read_readvariableop1savev2_adam_residual_1_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop/savev2_adam_dense_11_bias_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableop7savev2_adam_class_output_1_kernel_v_read_readvariableop5savev2_adam_class_output_1_bias_v_read_readvariableop3savev2_adam_residual_1_kernel_v_read_readvariableop1savev2_adam_residual_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *K
dtypesA
?2=	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�@:@:@ : :	 �:�:
��:�:
��:�:
��:�:	� : : :: :: : : : : : : :	�@:@:@ : :	 �:�:
��:�:
��:�:
��:�:	� : : :: ::	�@:@:@ : :	 �:�:
��:�:
��:�:
��:�:	� : : :: :: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :%!

_output_shapes
:	 �:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&	"
 
_output_shapes
:
��:!


_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	� : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :%!

_output_shapes
:	 �:!

_output_shapes	
:�:& "
 
_output_shapes
:
��:!!

_output_shapes	
:�:&""
 
_output_shapes
:
��:!#

_output_shapes	
:�:&$"
 
_output_shapes
:
��:!%

_output_shapes	
:�:%&!

_output_shapes
:	� : '

_output_shapes
: :$( 

_output_shapes

: : )

_output_shapes
::$* 

_output_shapes

: : +

_output_shapes
::%,!

_output_shapes
:	�@: -

_output_shapes
:@:$. 

_output_shapes

:@ : /

_output_shapes
: :%0!

_output_shapes
:	 �:!1

_output_shapes	
:�:&2"
 
_output_shapes
:
��:!3

_output_shapes	
:�:&4"
 
_output_shapes
:
��:!5

_output_shapes	
:�:&6"
 
_output_shapes
:
��:!7

_output_shapes	
:�:%8!

_output_shapes
:	� : 9

_output_shapes
: :$: 

_output_shapes

: : ;

_output_shapes
::$< 

_output_shapes

: : =

_output_shapes
::>

_output_shapes
: 
�
�
H__inference_class_output_layer_call_and_return_conditional_losses_700153

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :::O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
D__inference_residual_layer_call_and_return_conditional_losses_700172

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :::O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
C__inference_dense_7_layer_call_and_return_conditional_losses_700013

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
Z
.__inference_concatenate_1_layer_call_fn_700194
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_6994902
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
~
)__inference_dense_10_layer_call_fn_700082

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_6993332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
D__inference_dense_10_layer_call_and_return_conditional_losses_700073

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�4
�
C__inference_model_1_layer_call_and_return_conditional_losses_699550
input_2
dense_7_699503
dense_7_699505
dense_8_699508
dense_8_699510
dense_9_699513
dense_9_699515
dense_10_699518
dense_10_699520
dense_11_699523
dense_11_699525
dense_12_699528
dense_12_699530
dense_13_699533
dense_13_699535
class_output_699538
class_output_699540
residual_699543
residual_699545
identity��$class_output/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall� residual/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_7_699503dense_7_699505*
Tin
2*
Tout
2*'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_6992522!
dense_7/StatefulPartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_699508dense_8_699510*
Tin
2*
Tout
2*'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_6992792!
dense_8/StatefulPartitionedCall�
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_699513dense_9_699515*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_6993062!
dense_9/StatefulPartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_699518dense_10_699520*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_6993332"
 dense_10/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_699523dense_11_699525*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_6993602"
 dense_11/StatefulPartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_699528dense_12_699530*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_6993872"
 dense_12/StatefulPartitionedCall�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_699533dense_13_699535*
Tin
2*
Tout
2*'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_6994142"
 dense_13/StatefulPartitionedCall�
$class_output/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0class_output_699538class_output_699540*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_class_output_layer_call_and_return_conditional_losses_6994412&
$class_output/StatefulPartitionedCall�
 residual/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0residual_699543residual_699545*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_residual_layer_call_and_return_conditional_losses_6994672"
 residual/StatefulPartitionedCall�
concatenate_1/PartitionedCallPartitionedCall-class_output/StatefulPartitionedCall:output:0)residual/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_6994902
concatenate_1/PartitionedCall�
IdentityIdentity&concatenate_1/PartitionedCall:output:0%^class_output/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall!^residual/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:����������::::::::::::::::::2L
$class_output/StatefulPartitionedCall$class_output/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2D
 residual/StatefulPartitionedCall residual/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
}
(__inference_dense_7_layer_call_fn_700022

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_6992522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
}
(__inference_dense_8_layer_call_fn_700042

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_6992792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
}
(__inference_dense_9_layer_call_fn_700062

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_6993062
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
C__inference_dense_8_layer_call_and_return_conditional_losses_699279

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:::O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�R
�
!__inference__wrapped_model_699237
input_22
.model_1_dense_7_matmul_readvariableop_resource3
/model_1_dense_7_biasadd_readvariableop_resource2
.model_1_dense_8_matmul_readvariableop_resource3
/model_1_dense_8_biasadd_readvariableop_resource2
.model_1_dense_9_matmul_readvariableop_resource3
/model_1_dense_9_biasadd_readvariableop_resource3
/model_1_dense_10_matmul_readvariableop_resource4
0model_1_dense_10_biasadd_readvariableop_resource3
/model_1_dense_11_matmul_readvariableop_resource4
0model_1_dense_11_biasadd_readvariableop_resource3
/model_1_dense_12_matmul_readvariableop_resource4
0model_1_dense_12_biasadd_readvariableop_resource3
/model_1_dense_13_matmul_readvariableop_resource4
0model_1_dense_13_biasadd_readvariableop_resource7
3model_1_class_output_matmul_readvariableop_resource8
4model_1_class_output_biasadd_readvariableop_resource3
/model_1_residual_matmul_readvariableop_resource4
0model_1_residual_biasadd_readvariableop_resource
identity��
%model_1/dense_7/MatMul/ReadVariableOpReadVariableOp.model_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02'
%model_1/dense_7/MatMul/ReadVariableOp�
model_1/dense_7/MatMulMatMulinput_2-model_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
model_1/dense_7/MatMul�
&model_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&model_1/dense_7/BiasAdd/ReadVariableOp�
model_1/dense_7/BiasAddBiasAdd model_1/dense_7/MatMul:product:0.model_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
model_1/dense_7/BiasAdd�
model_1/dense_7/ReluRelu model_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
model_1/dense_7/Relu�
%model_1/dense_8/MatMul/ReadVariableOpReadVariableOp.model_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02'
%model_1/dense_8/MatMul/ReadVariableOp�
model_1/dense_8/MatMulMatMul"model_1/dense_7/Relu:activations:0-model_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_8/MatMul�
&model_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&model_1/dense_8/BiasAdd/ReadVariableOp�
model_1/dense_8/BiasAddBiasAdd model_1/dense_8/MatMul:product:0.model_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_8/BiasAdd�
model_1/dense_8/ReluRelu model_1/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_8/Relu�
%model_1/dense_9/MatMul/ReadVariableOpReadVariableOp.model_1_dense_9_matmul_readvariableop_resource*
_output_shapes
:	 �*
dtype02'
%model_1/dense_9/MatMul/ReadVariableOp�
model_1/dense_9/MatMulMatMul"model_1/dense_8/Relu:activations:0-model_1/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_1/dense_9/MatMul�
&model_1/dense_9/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&model_1/dense_9/BiasAdd/ReadVariableOp�
model_1/dense_9/BiasAddBiasAdd model_1/dense_9/MatMul:product:0.model_1/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_1/dense_9/BiasAdd�
model_1/dense_9/ReluRelu model_1/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
model_1/dense_9/Relu�
&model_1/dense_10/MatMul/ReadVariableOpReadVariableOp/model_1_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02(
&model_1/dense_10/MatMul/ReadVariableOp�
model_1/dense_10/MatMulMatMul"model_1/dense_9/Relu:activations:0.model_1/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_1/dense_10/MatMul�
'model_1/dense_10/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'model_1/dense_10/BiasAdd/ReadVariableOp�
model_1/dense_10/BiasAddBiasAdd!model_1/dense_10/MatMul:product:0/model_1/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_1/dense_10/BiasAdd�
model_1/dense_10/ReluRelu!model_1/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
model_1/dense_10/Relu�
&model_1/dense_11/MatMul/ReadVariableOpReadVariableOp/model_1_dense_11_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02(
&model_1/dense_11/MatMul/ReadVariableOp�
model_1/dense_11/MatMulMatMul#model_1/dense_10/Relu:activations:0.model_1/dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_1/dense_11/MatMul�
'model_1/dense_11/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'model_1/dense_11/BiasAdd/ReadVariableOp�
model_1/dense_11/BiasAddBiasAdd!model_1/dense_11/MatMul:product:0/model_1/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_1/dense_11/BiasAdd�
model_1/dense_11/ReluRelu!model_1/dense_11/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
model_1/dense_11/Relu�
&model_1/dense_12/MatMul/ReadVariableOpReadVariableOp/model_1_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02(
&model_1/dense_12/MatMul/ReadVariableOp�
model_1/dense_12/MatMulMatMul#model_1/dense_11/Relu:activations:0.model_1/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_1/dense_12/MatMul�
'model_1/dense_12/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'model_1/dense_12/BiasAdd/ReadVariableOp�
model_1/dense_12/BiasAddBiasAdd!model_1/dense_12/MatMul:product:0/model_1/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_1/dense_12/BiasAdd�
model_1/dense_12/ReluRelu!model_1/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
model_1/dense_12/Relu�
&model_1/dense_13/MatMul/ReadVariableOpReadVariableOp/model_1_dense_13_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype02(
&model_1/dense_13/MatMul/ReadVariableOp�
model_1/dense_13/MatMulMatMul#model_1/dense_12/Relu:activations:0.model_1/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_13/MatMul�
'model_1/dense_13/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_1/dense_13/BiasAdd/ReadVariableOp�
model_1/dense_13/BiasAddBiasAdd!model_1/dense_13/MatMul:product:0/model_1/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_13/BiasAdd�
model_1/dense_13/ReluRelu!model_1/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_13/Relu�
*model_1/class_output/MatMul/ReadVariableOpReadVariableOp3model_1_class_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*model_1/class_output/MatMul/ReadVariableOp�
model_1/class_output/MatMulMatMul#model_1/dense_13/Relu:activations:02model_1/class_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_1/class_output/MatMul�
+model_1/class_output/BiasAdd/ReadVariableOpReadVariableOp4model_1_class_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_1/class_output/BiasAdd/ReadVariableOp�
model_1/class_output/BiasAddBiasAdd%model_1/class_output/MatMul:product:03model_1/class_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_1/class_output/BiasAdd�
model_1/class_output/SoftmaxSoftmax%model_1/class_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
model_1/class_output/Softmax�
&model_1/residual/MatMul/ReadVariableOpReadVariableOp/model_1_residual_matmul_readvariableop_resource*
_output_shapes

: *
dtype02(
&model_1/residual/MatMul/ReadVariableOp�
model_1/residual/MatMulMatMul#model_1/dense_13/Relu:activations:0.model_1/residual/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_1/residual/MatMul�
'model_1/residual/BiasAdd/ReadVariableOpReadVariableOp0model_1_residual_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/residual/BiasAdd/ReadVariableOp�
model_1/residual/BiasAddBiasAdd!model_1/residual/MatMul:product:0/model_1/residual/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_1/residual/BiasAdd�
!model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_1/concat/axis�
model_1/concatenate_1/concatConcatV2&model_1/class_output/Softmax:softmax:0!model_1/residual/BiasAdd:output:0*model_1/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
model_1/concatenate_1/concaty
IdentityIdentity%model_1/concatenate_1/concat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:����������:::::::::::::::::::Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�G
�
C__inference_model_1_layer_call_and_return_conditional_losses_699852

inputs*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource/
+class_output_matmul_readvariableop_resource0
,class_output_biasadd_readvariableop_resource+
'residual_matmul_readvariableop_resource,
(residual_biasadd_readvariableop_resource
identity��
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
dense_7/MatMul/ReadVariableOp�
dense_7/MatMulMatMulinputs%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_7/MatMul�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_7/BiasAdd/ReadVariableOp�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_7/BiasAddp
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_7/Relu�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMuldense_7/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_8/MatMul�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_8/BiasAdd/ReadVariableOp�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_8/BiasAddp
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_8/Relu�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	 �*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_9/Relu�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/MatMul�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_10/BiasAdd/ReadVariableOp�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_10/Relu�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_11/MatMul�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_11/BiasAdd/ReadVariableOp�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_11/BiasAddt
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_11/Relu�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMuldense_11/Relu:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/BiasAddt
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_12/Relu�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_13/MatMul�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_13/BiasAdd/ReadVariableOp�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_13/BiasAdds
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_13/Relu�
"class_output/MatMul/ReadVariableOpReadVariableOp+class_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"class_output/MatMul/ReadVariableOp�
class_output/MatMulMatMuldense_13/Relu:activations:0*class_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
class_output/MatMul�
#class_output/BiasAdd/ReadVariableOpReadVariableOp,class_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#class_output/BiasAdd/ReadVariableOp�
class_output/BiasAddBiasAddclass_output/MatMul:product:0+class_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
class_output/BiasAdd�
class_output/SoftmaxSoftmaxclass_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
class_output/Softmax�
residual/MatMul/ReadVariableOpReadVariableOp'residual_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
residual/MatMul/ReadVariableOp�
residual/MatMulMatMuldense_13/Relu:activations:0&residual/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
residual/MatMul�
residual/BiasAdd/ReadVariableOpReadVariableOp(residual_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
residual/BiasAdd/ReadVariableOp�
residual/BiasAddBiasAddresidual/MatMul:product:0'residual/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
residual/BiasAddx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis�
concatenate_1/concatConcatV2class_output/Softmax:softmax:0residual/BiasAdd:output:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatenate_1/concatq
IdentityIdentityconcatenate_1/concat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:����������:::::::::::::::::::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�G
�
C__inference_model_1_layer_call_and_return_conditional_losses_699920

inputs*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource/
+class_output_matmul_readvariableop_resource0
,class_output_biasadd_readvariableop_resource+
'residual_matmul_readvariableop_resource,
(residual_biasadd_readvariableop_resource
identity��
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
dense_7/MatMul/ReadVariableOp�
dense_7/MatMulMatMulinputs%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_7/MatMul�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_7/BiasAdd/ReadVariableOp�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_7/BiasAddp
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_7/Relu�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMuldense_7/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_8/MatMul�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_8/BiasAdd/ReadVariableOp�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_8/BiasAddp
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_8/Relu�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	 �*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/BiasAddq
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_9/Relu�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/MatMul�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_10/BiasAdd/ReadVariableOp�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_10/Relu�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_11/MatMul�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_11/BiasAdd/ReadVariableOp�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_11/BiasAddt
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_11/Relu�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMuldense_11/Relu:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/BiasAddt
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_12/Relu�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_13/MatMul�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_13/BiasAdd/ReadVariableOp�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_13/BiasAdds
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_13/Relu�
"class_output/MatMul/ReadVariableOpReadVariableOp+class_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"class_output/MatMul/ReadVariableOp�
class_output/MatMulMatMuldense_13/Relu:activations:0*class_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
class_output/MatMul�
#class_output/BiasAdd/ReadVariableOpReadVariableOp,class_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#class_output/BiasAdd/ReadVariableOp�
class_output/BiasAddBiasAddclass_output/MatMul:product:0+class_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
class_output/BiasAdd�
class_output/SoftmaxSoftmaxclass_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
class_output/Softmax�
residual/MatMul/ReadVariableOpReadVariableOp'residual_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
residual/MatMul/ReadVariableOp�
residual/MatMulMatMuldense_13/Relu:activations:0&residual/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
residual/MatMul�
residual/BiasAdd/ReadVariableOpReadVariableOp(residual_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
residual/BiasAdd/ReadVariableOp�
residual/BiasAddBiasAddresidual/MatMul:product:0'residual/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
residual/BiasAddx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis�
concatenate_1/concatConcatV2class_output/Softmax:softmax:0residual/BiasAdd:output:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatenate_1/concatq
IdentityIdentityconcatenate_1/concat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:����������:::::::::::::::::::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_model_1_layer_call_fn_699642
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_6996032
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:����������::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
C__inference_dense_9_layer_call_and_return_conditional_losses_699306

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 �*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :::O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�4
�
C__inference_model_1_layer_call_and_return_conditional_losses_699603

inputs
dense_7_699556
dense_7_699558
dense_8_699561
dense_8_699563
dense_9_699566
dense_9_699568
dense_10_699571
dense_10_699573
dense_11_699576
dense_11_699578
dense_12_699581
dense_12_699583
dense_13_699586
dense_13_699588
class_output_699591
class_output_699593
residual_699596
residual_699598
identity��$class_output/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall� residual/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_699556dense_7_699558*
Tin
2*
Tout
2*'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_6992522!
dense_7/StatefulPartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_699561dense_8_699563*
Tin
2*
Tout
2*'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_6992792!
dense_8/StatefulPartitionedCall�
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_699566dense_9_699568*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_6993062!
dense_9/StatefulPartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_699571dense_10_699573*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_6993332"
 dense_10/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_699576dense_11_699578*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_6993602"
 dense_11/StatefulPartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_699581dense_12_699583*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_6993872"
 dense_12/StatefulPartitionedCall�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_699586dense_13_699588*
Tin
2*
Tout
2*'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_6994142"
 dense_13/StatefulPartitionedCall�
$class_output/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0class_output_699591class_output_699593*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_class_output_layer_call_and_return_conditional_losses_6994412&
$class_output/StatefulPartitionedCall�
 residual/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0residual_699596residual_699598*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_residual_layer_call_and_return_conditional_losses_6994672"
 residual/StatefulPartitionedCall�
concatenate_1/PartitionedCallPartitionedCall-class_output/StatefulPartitionedCall:output:0)residual/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_6994902
concatenate_1/PartitionedCall�
IdentityIdentity&concatenate_1/PartitionedCall:output:0%^class_output/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall!^residual/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:����������::::::::::::::::::2L
$class_output/StatefulPartitionedCall$class_output/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2D
 residual/StatefulPartitionedCall residual/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
~
)__inference_dense_13_layer_call_fn_700142

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_6994142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
D__inference_dense_11_layer_call_and_return_conditional_losses_700093

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
C__inference_dense_8_layer_call_and_return_conditional_losses_700033

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:::O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�4
�
C__inference_model_1_layer_call_and_return_conditional_losses_699694

inputs
dense_7_699647
dense_7_699649
dense_8_699652
dense_8_699654
dense_9_699657
dense_9_699659
dense_10_699662
dense_10_699664
dense_11_699667
dense_11_699669
dense_12_699672
dense_12_699674
dense_13_699677
dense_13_699679
class_output_699682
class_output_699684
residual_699687
residual_699689
identity��$class_output/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall� residual/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_699647dense_7_699649*
Tin
2*
Tout
2*'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_6992522!
dense_7/StatefulPartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_699652dense_8_699654*
Tin
2*
Tout
2*'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_6992792!
dense_8/StatefulPartitionedCall�
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_699657dense_9_699659*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_6993062!
dense_9/StatefulPartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_699662dense_10_699664*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_6993332"
 dense_10/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_699667dense_11_699669*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_6993602"
 dense_11/StatefulPartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_699672dense_12_699674*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_6993872"
 dense_12/StatefulPartitionedCall�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_699677dense_13_699679*
Tin
2*
Tout
2*'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_6994142"
 dense_13/StatefulPartitionedCall�
$class_output/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0class_output_699682class_output_699684*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_class_output_layer_call_and_return_conditional_losses_6994412&
$class_output/StatefulPartitionedCall�
 residual/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0residual_699687residual_699689*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_residual_layer_call_and_return_conditional_losses_6994672"
 residual/StatefulPartitionedCall�
concatenate_1/PartitionedCallPartitionedCall-class_output/StatefulPartitionedCall:output:0)residual/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_6994902
concatenate_1/PartitionedCall�
IdentityIdentity&concatenate_1/PartitionedCall:output:0%^class_output/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall!^residual/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:����������::::::::::::::::::2L
$class_output/StatefulPartitionedCall$class_output/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2D
 residual/StatefulPartitionedCall residual/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
C__inference_dense_7_layer_call_and_return_conditional_losses_699252

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
��
� 
"__inference__traced_restore_700599
file_prefix#
assignvariableop_dense_7_kernel#
assignvariableop_1_dense_7_bias%
!assignvariableop_2_dense_8_kernel#
assignvariableop_3_dense_8_bias%
!assignvariableop_4_dense_9_kernel#
assignvariableop_5_dense_9_bias&
"assignvariableop_6_dense_10_kernel$
 assignvariableop_7_dense_10_bias&
"assignvariableop_8_dense_11_kernel$
 assignvariableop_9_dense_11_bias'
#assignvariableop_10_dense_12_kernel%
!assignvariableop_11_dense_12_bias'
#assignvariableop_12_dense_13_kernel%
!assignvariableop_13_dense_13_bias-
)assignvariableop_14_class_output_1_kernel+
'assignvariableop_15_class_output_1_bias)
%assignvariableop_16_residual_1_kernel'
#assignvariableop_17_residual_1_bias!
assignvariableop_18_adam_iter#
assignvariableop_19_adam_beta_1#
assignvariableop_20_adam_beta_2"
assignvariableop_21_adam_decay*
&assignvariableop_22_adam_learning_rate
assignvariableop_23_total
assignvariableop_24_count-
)assignvariableop_25_adam_dense_7_kernel_m+
'assignvariableop_26_adam_dense_7_bias_m-
)assignvariableop_27_adam_dense_8_kernel_m+
'assignvariableop_28_adam_dense_8_bias_m-
)assignvariableop_29_adam_dense_9_kernel_m+
'assignvariableop_30_adam_dense_9_bias_m.
*assignvariableop_31_adam_dense_10_kernel_m,
(assignvariableop_32_adam_dense_10_bias_m.
*assignvariableop_33_adam_dense_11_kernel_m,
(assignvariableop_34_adam_dense_11_bias_m.
*assignvariableop_35_adam_dense_12_kernel_m,
(assignvariableop_36_adam_dense_12_bias_m.
*assignvariableop_37_adam_dense_13_kernel_m,
(assignvariableop_38_adam_dense_13_bias_m4
0assignvariableop_39_adam_class_output_1_kernel_m2
.assignvariableop_40_adam_class_output_1_bias_m0
,assignvariableop_41_adam_residual_1_kernel_m.
*assignvariableop_42_adam_residual_1_bias_m-
)assignvariableop_43_adam_dense_7_kernel_v+
'assignvariableop_44_adam_dense_7_bias_v-
)assignvariableop_45_adam_dense_8_kernel_v+
'assignvariableop_46_adam_dense_8_bias_v-
)assignvariableop_47_adam_dense_9_kernel_v+
'assignvariableop_48_adam_dense_9_bias_v.
*assignvariableop_49_adam_dense_10_kernel_v,
(assignvariableop_50_adam_dense_10_bias_v.
*assignvariableop_51_adam_dense_11_kernel_v,
(assignvariableop_52_adam_dense_11_bias_v.
*assignvariableop_53_adam_dense_12_kernel_v,
(assignvariableop_54_adam_dense_12_bias_v.
*assignvariableop_55_adam_dense_13_kernel_v,
(assignvariableop_56_adam_dense_13_bias_v4
0assignvariableop_57_adam_class_output_1_kernel_v2
.assignvariableop_58_adam_class_output_1_bias_v0
,assignvariableop_59_adam_residual_1_kernel_v.
*assignvariableop_60_adam_residual_1_bias_v
identity_62��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�!
value�!B�!=B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*K
dtypesA
?2=	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_7_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_7_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_8_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_8_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_9_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_9_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_10_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_10_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_11_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_11_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_12_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_12_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_13_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_13_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp)assignvariableop_14_class_output_1_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp'assignvariableop_15_class_output_1_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp%assignvariableop_16_residual_1_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp#assignvariableop_17_residual_1_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0	*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_7_kernel_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_7_bias_mIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_8_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_8_bias_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_9_kernel_mIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_9_bias_mIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_10_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_10_bias_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_11_kernel_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_11_bias_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_12_kernel_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_12_bias_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_13_kernel_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_13_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp0assignvariableop_39_adam_class_output_1_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp.assignvariableop_40_adam_class_output_1_bias_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adam_residual_1_kernel_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_residual_1_bias_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_7_kernel_vIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_7_bias_vIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_8_kernel_vIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_8_bias_vIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_9_kernel_vIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_9_bias_vIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_10_kernel_vIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_10_bias_vIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_11_kernel_vIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_11_bias_vIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_12_kernel_vIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_12_bias_vIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_13_kernel_vIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_13_bias_vIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOp0assignvariableop_57_adam_class_output_1_kernel_vIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOp.assignvariableop_58_adam_class_output_1_bias_vIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOp,assignvariableop_59_adam_residual_1_kernel_vIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp*assignvariableop_60_adam_residual_1_bias_vIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_61Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_61�
Identity_62IdentityIdentity_61:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_62"#
identity_62Identity_62:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: 
�
�
D__inference_dense_10_layer_call_and_return_conditional_losses_699333

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
<
input_21
serving_default_input_2:0����������A
concatenate_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�X
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer-10
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
�_default_save_signature
�__call__
+�&call_and_return_all_conditional_losses"�S
_tf_keras_model�S{"class_name": "Model", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 148]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["dense_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["dense_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["dense_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "class_output", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "class_output", "inbound_nodes": [[["dense_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "residual", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "residual", "inbound_nodes": [[["dense_13", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["class_output", 0, 0, {}], ["residual", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["concatenate_1", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 148]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 148]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["dense_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["dense_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["dense_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "class_output", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "class_output", "inbound_nodes": [[["dense_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "residual", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "residual", "inbound_nodes": [[["dense_13", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["class_output", 0, 0, {}], ["residual", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["concatenate_1", 0, 0]]}}, "training_config": {"loss": "custom_mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 148]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 148]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 148}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 148]}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
�

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
�

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
�

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
�

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
�

6kernel
7bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
�

<kernel
=bias
>trainable_variables
?regularization_losses
@	variables
A	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "class_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "class_output", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
�

Bkernel
Cbias
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "residual", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "residual", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
�
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 3]}, {"class_name": "TensorShape", "items": [null, 3]}]}
�
Liter

Mbeta_1

Nbeta_2
	Odecay
Plearning_ratem�m�m�m�m�m�$m�%m�*m�+m�0m�1m�6m�7m�<m�=m�Bm�Cm�v�v�v�v�v�v�$v�%v�*v�+v�0v�1v�6v�7v�<v�=v�Bv�Cv�"
	optimizer
�
0
1
2
3
4
5
$6
%7
*8
+9
010
111
612
713
<14
=15
B16
C17"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
2
3
4
5
$6
%7
*8
+9
010
111
612
713
<14
=15
B16
C17"
trackable_list_wrapper
�
trainable_variables
Qmetrics
Rnon_trainable_variables

Slayers
regularization_losses
Tlayer_regularization_losses
	variables
Ulayer_metrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
!:	�@2dense_7/kernel
:@2dense_7/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
trainable_variables
Vmetrics
Wnon_trainable_variables

Xlayers
regularization_losses
Ylayer_regularization_losses
	variables
Zlayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :@ 2dense_8/kernel
: 2dense_8/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
trainable_variables
[metrics
\non_trainable_variables

]layers
regularization_losses
^layer_regularization_losses
	variables
_layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	 �2dense_9/kernel
:�2dense_9/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
 trainable_variables
`metrics
anon_trainable_variables

blayers
!regularization_losses
clayer_regularization_losses
"	variables
dlayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!
��2dense_10/kernel
:�2dense_10/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
�
&trainable_variables
emetrics
fnon_trainable_variables

glayers
'regularization_losses
hlayer_regularization_losses
(	variables
ilayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!
��2dense_11/kernel
:�2dense_11/bias
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
�
,trainable_variables
jmetrics
knon_trainable_variables

llayers
-regularization_losses
mlayer_regularization_losses
.	variables
nlayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!
��2dense_12/kernel
:�2dense_12/bias
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
�
2trainable_variables
ometrics
pnon_trainable_variables

qlayers
3regularization_losses
rlayer_regularization_losses
4	variables
slayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 	� 2dense_13/kernel
: 2dense_13/bias
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
�
8trainable_variables
tmetrics
unon_trainable_variables

vlayers
9regularization_losses
wlayer_regularization_losses
:	variables
xlayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':% 2class_output_1/kernel
!:2class_output_1/bias
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
�
>trainable_variables
ymetrics
znon_trainable_variables

{layers
?regularization_losses
|layer_regularization_losses
@	variables
}layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:! 2residual_1/kernel
:2residual_1/bias
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
�
Dtrainable_variables
~metrics
non_trainable_variables
�layers
Eregularization_losses
 �layer_regularization_losses
F	variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Htrainable_variables
�metrics
�non_trainable_variables
�layers
Iregularization_losses
 �layer_regularization_losses
J	variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
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
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
&:$	�@2Adam/dense_7/kernel/m
:@2Adam/dense_7/bias/m
%:#@ 2Adam/dense_8/kernel/m
: 2Adam/dense_8/bias/m
&:$	 �2Adam/dense_9/kernel/m
 :�2Adam/dense_9/bias/m
(:&
��2Adam/dense_10/kernel/m
!:�2Adam/dense_10/bias/m
(:&
��2Adam/dense_11/kernel/m
!:�2Adam/dense_11/bias/m
(:&
��2Adam/dense_12/kernel/m
!:�2Adam/dense_12/bias/m
':%	� 2Adam/dense_13/kernel/m
 : 2Adam/dense_13/bias/m
,:* 2Adam/class_output_1/kernel/m
&:$2Adam/class_output_1/bias/m
(:& 2Adam/residual_1/kernel/m
": 2Adam/residual_1/bias/m
&:$	�@2Adam/dense_7/kernel/v
:@2Adam/dense_7/bias/v
%:#@ 2Adam/dense_8/kernel/v
: 2Adam/dense_8/bias/v
&:$	 �2Adam/dense_9/kernel/v
 :�2Adam/dense_9/bias/v
(:&
��2Adam/dense_10/kernel/v
!:�2Adam/dense_10/bias/v
(:&
��2Adam/dense_11/kernel/v
!:�2Adam/dense_11/bias/v
(:&
��2Adam/dense_12/kernel/v
!:�2Adam/dense_12/bias/v
':%	� 2Adam/dense_13/kernel/v
 : 2Adam/dense_13/bias/v
,:* 2Adam/class_output_1/kernel/v
&:$2Adam/class_output_1/bias/v
(:& 2Adam/residual_1/kernel/v
": 2Adam/residual_1/bias/v
�2�
!__inference__wrapped_model_699237�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *'�$
"�
input_2����������
�2�
(__inference_model_1_layer_call_fn_699733
(__inference_model_1_layer_call_fn_699961
(__inference_model_1_layer_call_fn_700002
(__inference_model_1_layer_call_fn_699642�
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
C__inference_model_1_layer_call_and_return_conditional_losses_699852
C__inference_model_1_layer_call_and_return_conditional_losses_699920
C__inference_model_1_layer_call_and_return_conditional_losses_699550
C__inference_model_1_layer_call_and_return_conditional_losses_699500�
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
�2�
(__inference_dense_7_layer_call_fn_700022�
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
C__inference_dense_7_layer_call_and_return_conditional_losses_700013�
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
(__inference_dense_8_layer_call_fn_700042�
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
C__inference_dense_8_layer_call_and_return_conditional_losses_700033�
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
(__inference_dense_9_layer_call_fn_700062�
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
C__inference_dense_9_layer_call_and_return_conditional_losses_700053�
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
)__inference_dense_10_layer_call_fn_700082�
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
D__inference_dense_10_layer_call_and_return_conditional_losses_700073�
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
)__inference_dense_11_layer_call_fn_700102�
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
D__inference_dense_11_layer_call_and_return_conditional_losses_700093�
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
)__inference_dense_12_layer_call_fn_700122�
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
D__inference_dense_12_layer_call_and_return_conditional_losses_700113�
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
)__inference_dense_13_layer_call_fn_700142�
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
D__inference_dense_13_layer_call_and_return_conditional_losses_700133�
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
-__inference_class_output_layer_call_fn_700162�
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
H__inference_class_output_layer_call_and_return_conditional_losses_700153�
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
)__inference_residual_layer_call_fn_700181�
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
D__inference_residual_layer_call_and_return_conditional_losses_700172�
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
.__inference_concatenate_1_layer_call_fn_700194�
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
I__inference_concatenate_1_layer_call_and_return_conditional_losses_700188�
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
3B1
$__inference_signature_wrapper_699784input_2�
!__inference__wrapped_model_699237�$%*+0167<=BC1�.
'�$
"�
input_2����������
� "=�:
8
concatenate_1'�$
concatenate_1����������
H__inference_class_output_layer_call_and_return_conditional_losses_700153\<=/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� �
-__inference_class_output_layer_call_fn_700162O<=/�,
%�"
 �
inputs��������� 
� "�����������
I__inference_concatenate_1_layer_call_and_return_conditional_losses_700188�Z�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "%�"
�
0���������
� �
.__inference_concatenate_1_layer_call_fn_700194vZ�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "�����������
D__inference_dense_10_layer_call_and_return_conditional_losses_700073^$%0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
)__inference_dense_10_layer_call_fn_700082Q$%0�-
&�#
!�
inputs����������
� "������������
D__inference_dense_11_layer_call_and_return_conditional_losses_700093^*+0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
)__inference_dense_11_layer_call_fn_700102Q*+0�-
&�#
!�
inputs����������
� "������������
D__inference_dense_12_layer_call_and_return_conditional_losses_700113^010�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
)__inference_dense_12_layer_call_fn_700122Q010�-
&�#
!�
inputs����������
� "������������
D__inference_dense_13_layer_call_and_return_conditional_losses_700133]670�-
&�#
!�
inputs����������
� "%�"
�
0��������� 
� }
)__inference_dense_13_layer_call_fn_700142P670�-
&�#
!�
inputs����������
� "���������� �
C__inference_dense_7_layer_call_and_return_conditional_losses_700013]0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� |
(__inference_dense_7_layer_call_fn_700022P0�-
&�#
!�
inputs����������
� "����������@�
C__inference_dense_8_layer_call_and_return_conditional_losses_700033\/�,
%�"
 �
inputs���������@
� "%�"
�
0��������� 
� {
(__inference_dense_8_layer_call_fn_700042O/�,
%�"
 �
inputs���������@
� "���������� �
C__inference_dense_9_layer_call_and_return_conditional_losses_700053]/�,
%�"
 �
inputs��������� 
� "&�#
�
0����������
� |
(__inference_dense_9_layer_call_fn_700062P/�,
%�"
 �
inputs��������� 
� "������������
C__inference_model_1_layer_call_and_return_conditional_losses_699500v$%*+0167<=BC9�6
/�,
"�
input_2����������
p

 
� "%�"
�
0���������
� �
C__inference_model_1_layer_call_and_return_conditional_losses_699550v$%*+0167<=BC9�6
/�,
"�
input_2����������
p 

 
� "%�"
�
0���������
� �
C__inference_model_1_layer_call_and_return_conditional_losses_699852u$%*+0167<=BC8�5
.�+
!�
inputs����������
p

 
� "%�"
�
0���������
� �
C__inference_model_1_layer_call_and_return_conditional_losses_699920u$%*+0167<=BC8�5
.�+
!�
inputs����������
p 

 
� "%�"
�
0���������
� �
(__inference_model_1_layer_call_fn_699642i$%*+0167<=BC9�6
/�,
"�
input_2����������
p

 
� "�����������
(__inference_model_1_layer_call_fn_699733i$%*+0167<=BC9�6
/�,
"�
input_2����������
p 

 
� "�����������
(__inference_model_1_layer_call_fn_699961h$%*+0167<=BC8�5
.�+
!�
inputs����������
p

 
� "�����������
(__inference_model_1_layer_call_fn_700002h$%*+0167<=BC8�5
.�+
!�
inputs����������
p 

 
� "�����������
D__inference_residual_layer_call_and_return_conditional_losses_700172\BC/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� |
)__inference_residual_layer_call_fn_700181OBC/�,
%�"
 �
inputs��������� 
� "�����������
$__inference_signature_wrapper_699784�$%*+0167<=BC<�9
� 
2�/
-
input_2"�
input_2����������"=�:
8
concatenate_1'�$
concatenate_1���������