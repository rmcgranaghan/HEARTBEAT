я
Њ§
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
dtypetype
О
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
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8ьЗ
{
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_namedense_21/kernel
t
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes
:	@*
dtype0
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:@*
dtype0
z
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_22/kernel
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

:@ *
dtype0
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
: *
dtype0
{
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 * 
shared_namedense_23/kernel
t
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes
:	 *
dtype0
s
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_23/bias
l
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes	
:*
dtype0
|
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_24/kernel
u
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel* 
_output_shapes
:
*
dtype0
s
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_24/bias
l
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes	
:*
dtype0
|
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_25/kernel
u
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel* 
_output_shapes
:
*
dtype0
s
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_25/bias
l
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes	
:*
dtype0
|
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_26/kernel
u
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel* 
_output_shapes
:
*
dtype0
s
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_26/bias
l
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes	
:*
dtype0
{
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 * 
shared_namedense_27/kernel
t
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes
:	 *
dtype0
r
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_27/bias
k
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes
: *
dtype0

class_output_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameclass_output_3/kernel

)class_output_3/kernel/Read/ReadVariableOpReadVariableOpclass_output_3/kernel*
_output_shapes

: *
dtype0
~
class_output_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameclass_output_3/bias
w
'class_output_3/bias/Read/ReadVariableOpReadVariableOpclass_output_3/bias*
_output_shapes
:*
dtype0
~
residual_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_nameresidual_3/kernel
w
%residual_3/kernel/Read/ReadVariableOpReadVariableOpresidual_3/kernel*
_output_shapes

: *
dtype0
v
residual_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameresidual_3/bias
o
#residual_3/bias/Read/ReadVariableOpReadVariableOpresidual_3/bias*
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

Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/dense_21/kernel/m

*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m*
_output_shapes
:	@*
dtype0

Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_21/bias/m
y
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_22/kernel/m

*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m*
_output_shapes

:@ *
dtype0

Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_22/bias/m
y
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes
: *
dtype0

Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *'
shared_nameAdam/dense_23/kernel/m

*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
_output_shapes
:	 *
dtype0

Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/m
z
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_24/kernel/m

*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_24/bias/m
z
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_25/kernel/m

*Adam/dense_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/m
z
(Adam/dense_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_26/kernel/m

*Adam/dense_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_26/bias/m
z
(Adam/dense_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *'
shared_nameAdam/dense_27/kernel/m

*Adam/dense_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/m*
_output_shapes
:	 *
dtype0

Adam/dense_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_27/bias/m
y
(Adam/dense_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/m*
_output_shapes
: *
dtype0

Adam/class_output_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_nameAdam/class_output_3/kernel/m

0Adam/class_output_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/class_output_3/kernel/m*
_output_shapes

: *
dtype0

Adam/class_output_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/class_output_3/bias/m

.Adam/class_output_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/class_output_3/bias/m*
_output_shapes
:*
dtype0

Adam/residual_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_nameAdam/residual_3/kernel/m

,Adam/residual_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/residual_3/kernel/m*
_output_shapes

: *
dtype0

Adam/residual_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/residual_3/bias/m
}
*Adam/residual_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/residual_3/bias/m*
_output_shapes
:*
dtype0

Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/dense_21/kernel/v

*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v*
_output_shapes
:	@*
dtype0

Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_21/bias/v
y
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_22/kernel/v

*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v*
_output_shapes

:@ *
dtype0

Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_22/bias/v
y
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes
: *
dtype0

Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *'
shared_nameAdam/dense_23/kernel/v

*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes
:	 *
dtype0

Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/v
z
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_24/kernel/v

*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_24/bias/v
z
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_25/kernel/v

*Adam/dense_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/v
z
(Adam/dense_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_26/kernel/v

*Adam/dense_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_26/bias/v
z
(Adam/dense_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *'
shared_nameAdam/dense_27/kernel/v

*Adam/dense_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/v*
_output_shapes
:	 *
dtype0

Adam/dense_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_27/bias/v
y
(Adam/dense_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/v*
_output_shapes
: *
dtype0

Adam/class_output_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_nameAdam/class_output_3/kernel/v

0Adam/class_output_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/class_output_3/kernel/v*
_output_shapes

: *
dtype0

Adam/class_output_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/class_output_3/bias/v

.Adam/class_output_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/class_output_3/bias/v*
_output_shapes
:*
dtype0

Adam/residual_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_nameAdam/residual_3/kernel/v

,Adam/residual_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/residual_3/kernel/v*
_output_shapes

: *
dtype0

Adam/residual_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/residual_3/bias/v
}
*Adam/residual_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/residual_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Ф]
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*џ\
valueѕ\Bђ\ Bы\
љ
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
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer-11
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
 bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
h

%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
h

+kernel
,bias
-trainable_variables
.regularization_losses
/	variables
0	keras_api
R
1trainable_variables
2regularization_losses
3	variables
4	keras_api
h

5kernel
6bias
7trainable_variables
8regularization_losses
9	variables
:	keras_api
h

;kernel
<bias
=trainable_variables
>regularization_losses
?	variables
@	keras_api
h

Akernel
Bbias
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
h

Gkernel
Hbias
Itrainable_variables
Jregularization_losses
K	variables
L	keras_api
R
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
Ј
Qiter

Rbeta_1

Sbeta_2
	Tdecay
Ulearning_ratemmmmm m%m&m+m,m 5mЁ6mЂ;mЃ<mЄAmЅBmІGmЇHmЈvЉvЊvЋvЌv­ vЎ%vЏ&vА+vБ,vВ5vГ6vД;vЕ<vЖAvЗBvИGvЙHvК

0
1
2
3
4
 5
%6
&7
+8
,9
510
611
;12
<13
A14
B15
G16
H17
 

0
1
2
3
4
 5
%6
&7
+8
,9
510
611
;12
<13
A14
B15
G16
H17
­
trainable_variables
Vmetrics
Wnon_trainable_variables

Xlayers
regularization_losses
Ylayer_regularization_losses
	variables
Zlayer_metrics
 
[Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_21/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
trainable_variables
[metrics
\non_trainable_variables

]layers
regularization_losses
^layer_regularization_losses
	variables
_layer_metrics
[Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_22/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
trainable_variables
`metrics
anon_trainable_variables

blayers
regularization_losses
clayer_regularization_losses
	variables
dlayer_metrics
[Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_23/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1
 

0
 1
­
!trainable_variables
emetrics
fnon_trainable_variables

glayers
"regularization_losses
hlayer_regularization_losses
#	variables
ilayer_metrics
[Y
VARIABLE_VALUEdense_24/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_24/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1
 

%0
&1
­
'trainable_variables
jmetrics
knon_trainable_variables

llayers
(regularization_losses
mlayer_regularization_losses
)	variables
nlayer_metrics
[Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_25/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
 

+0
,1
­
-trainable_variables
ometrics
pnon_trainable_variables

qlayers
.regularization_losses
rlayer_regularization_losses
/	variables
slayer_metrics
 
 
 
­
1trainable_variables
tmetrics
unon_trainable_variables

vlayers
2regularization_losses
wlayer_regularization_losses
3	variables
xlayer_metrics
[Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_26/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

50
61
 

50
61
­
7trainable_variables
ymetrics
znon_trainable_variables

{layers
8regularization_losses
|layer_regularization_losses
9	variables
}layer_metrics
[Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_27/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1
 

;0
<1
А
=trainable_variables
~metrics
non_trainable_variables
layers
>regularization_losses
 layer_regularization_losses
?	variables
layer_metrics
a_
VARIABLE_VALUEclass_output_3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEclass_output_3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

A0
B1
 

A0
B1
В
Ctrainable_variables
metrics
non_trainable_variables
layers
Dregularization_losses
 layer_regularization_losses
E	variables
layer_metrics
][
VARIABLE_VALUEresidual_3/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEresidual_3/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

G0
H1
 

G0
H1
В
Itrainable_variables
metrics
non_trainable_variables
layers
Jregularization_losses
 layer_regularization_losses
K	variables
layer_metrics
 
 
 
В
Mtrainable_variables
metrics
non_trainable_variables
layers
Nregularization_losses
 layer_regularization_losses
O	variables
layer_metrics
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
0
 
V
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
11
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
 
 
 
 
 
8

total

count
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
~|
VARIABLE_VALUEAdam/dense_21/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_21/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_22/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_22/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_23/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_23/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_24/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_24/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_25/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_26/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_26/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_27/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_27/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/class_output_3/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/class_output_3/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/residual_3/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/residual_3/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_21/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_21/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_22/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_22/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_23/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_23/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_24/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_24/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_25/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_26/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_26/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_27/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_27/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/class_output_3/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/class_output_3/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/residual_3/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/residual_3/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_4Placeholder*(
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
ш
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4dense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/biasdense_27/kerneldense_27/biasclass_output_3/kernelclass_output_3/biasresidual_3/kernelresidual_3/bias*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*.
f)R'
%__inference_signature_wrapper_2397951
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
н
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOp)class_output_3/kernel/Read/ReadVariableOp'class_output_3/bias/Read/ReadVariableOp%residual_3/kernel/Read/ReadVariableOp#residual_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_21/kernel/m/Read/ReadVariableOp(Adam/dense_21/bias/m/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp*Adam/dense_24/kernel/m/Read/ReadVariableOp(Adam/dense_24/bias/m/Read/ReadVariableOp*Adam/dense_25/kernel/m/Read/ReadVariableOp(Adam/dense_25/bias/m/Read/ReadVariableOp*Adam/dense_26/kernel/m/Read/ReadVariableOp(Adam/dense_26/bias/m/Read/ReadVariableOp*Adam/dense_27/kernel/m/Read/ReadVariableOp(Adam/dense_27/bias/m/Read/ReadVariableOp0Adam/class_output_3/kernel/m/Read/ReadVariableOp.Adam/class_output_3/bias/m/Read/ReadVariableOp,Adam/residual_3/kernel/m/Read/ReadVariableOp*Adam/residual_3/bias/m/Read/ReadVariableOp*Adam/dense_21/kernel/v/Read/ReadVariableOp(Adam/dense_21/bias/v/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOp*Adam/dense_24/kernel/v/Read/ReadVariableOp(Adam/dense_24/bias/v/Read/ReadVariableOp*Adam/dense_25/kernel/v/Read/ReadVariableOp(Adam/dense_25/bias/v/Read/ReadVariableOp*Adam/dense_26/kernel/v/Read/ReadVariableOp(Adam/dense_26/bias/v/Read/ReadVariableOp*Adam/dense_27/kernel/v/Read/ReadVariableOp(Adam/dense_27/bias/v/Read/ReadVariableOp0Adam/class_output_3/kernel/v/Read/ReadVariableOp.Adam/class_output_3/bias/v/Read/ReadVariableOp,Adam/residual_3/kernel/v/Read/ReadVariableOp*Adam/residual_3/bias/v/Read/ReadVariableOpConst*J
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
GPU2*0J 8*)
f$R"
 __inference__traced_save_2398607

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/biasdense_27/kerneldense_27/biasclass_output_3/kernelclass_output_3/biasresidual_3/kernelresidual_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_21/kernel/mAdam/dense_21/bias/mAdam/dense_22/kernel/mAdam/dense_22/bias/mAdam/dense_23/kernel/mAdam/dense_23/bias/mAdam/dense_24/kernel/mAdam/dense_24/bias/mAdam/dense_25/kernel/mAdam/dense_25/bias/mAdam/dense_26/kernel/mAdam/dense_26/bias/mAdam/dense_27/kernel/mAdam/dense_27/bias/mAdam/class_output_3/kernel/mAdam/class_output_3/bias/mAdam/residual_3/kernel/mAdam/residual_3/bias/mAdam/dense_21/kernel/vAdam/dense_21/bias/vAdam/dense_22/kernel/vAdam/dense_22/bias/vAdam/dense_23/kernel/vAdam/dense_23/bias/vAdam/dense_24/kernel/vAdam/dense_24/bias/vAdam/dense_25/kernel/vAdam/dense_25/bias/vAdam/dense_26/kernel/vAdam/dense_26/bias/vAdam/dense_27/kernel/vAdam/dense_27/bias/vAdam/class_output_3/kernel/vAdam/class_output_3/bias/vAdam/residual_3/kernel/vAdam/residual_3/bias/v*I
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
GPU2*0J 8*,
f'R%
#__inference__traced_restore_2398802чЙ	
я
­
E__inference_dense_24_layer_call_and_return_conditional_losses_2397467

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ћ
G
+__inference_dropout_1_layer_call_fn_2398305

inputs
identityІ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_23975272
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
я
­
E__inference_dense_26_layer_call_and_return_conditional_losses_2397551

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Р
Ё 
#__inference__traced_restore_2398802
file_prefix$
 assignvariableop_dense_21_kernel$
 assignvariableop_1_dense_21_bias&
"assignvariableop_2_dense_22_kernel$
 assignvariableop_3_dense_22_bias&
"assignvariableop_4_dense_23_kernel$
 assignvariableop_5_dense_23_bias&
"assignvariableop_6_dense_24_kernel$
 assignvariableop_7_dense_24_bias&
"assignvariableop_8_dense_25_kernel$
 assignvariableop_9_dense_25_bias'
#assignvariableop_10_dense_26_kernel%
!assignvariableop_11_dense_26_bias'
#assignvariableop_12_dense_27_kernel%
!assignvariableop_13_dense_27_bias-
)assignvariableop_14_class_output_3_kernel+
'assignvariableop_15_class_output_3_bias)
%assignvariableop_16_residual_3_kernel'
#assignvariableop_17_residual_3_bias!
assignvariableop_18_adam_iter#
assignvariableop_19_adam_beta_1#
assignvariableop_20_adam_beta_2"
assignvariableop_21_adam_decay*
&assignvariableop_22_adam_learning_rate
assignvariableop_23_total
assignvariableop_24_count.
*assignvariableop_25_adam_dense_21_kernel_m,
(assignvariableop_26_adam_dense_21_bias_m.
*assignvariableop_27_adam_dense_22_kernel_m,
(assignvariableop_28_adam_dense_22_bias_m.
*assignvariableop_29_adam_dense_23_kernel_m,
(assignvariableop_30_adam_dense_23_bias_m.
*assignvariableop_31_adam_dense_24_kernel_m,
(assignvariableop_32_adam_dense_24_bias_m.
*assignvariableop_33_adam_dense_25_kernel_m,
(assignvariableop_34_adam_dense_25_bias_m.
*assignvariableop_35_adam_dense_26_kernel_m,
(assignvariableop_36_adam_dense_26_bias_m.
*assignvariableop_37_adam_dense_27_kernel_m,
(assignvariableop_38_adam_dense_27_bias_m4
0assignvariableop_39_adam_class_output_3_kernel_m2
.assignvariableop_40_adam_class_output_3_bias_m0
,assignvariableop_41_adam_residual_3_kernel_m.
*assignvariableop_42_adam_residual_3_bias_m.
*assignvariableop_43_adam_dense_21_kernel_v,
(assignvariableop_44_adam_dense_21_bias_v.
*assignvariableop_45_adam_dense_22_kernel_v,
(assignvariableop_46_adam_dense_22_bias_v.
*assignvariableop_47_adam_dense_23_kernel_v,
(assignvariableop_48_adam_dense_23_bias_v.
*assignvariableop_49_adam_dense_24_kernel_v,
(assignvariableop_50_adam_dense_24_bias_v.
*assignvariableop_51_adam_dense_25_kernel_v,
(assignvariableop_52_adam_dense_25_bias_v.
*assignvariableop_53_adam_dense_26_kernel_v,
(assignvariableop_54_adam_dense_26_bias_v.
*assignvariableop_55_adam_dense_27_kernel_v,
(assignvariableop_56_adam_dense_27_bias_v4
0assignvariableop_57_adam_class_output_3_kernel_v2
.assignvariableop_58_adam_class_output_3_bias_v0
,assignvariableop_59_adam_residual_3_kernel_v.
*assignvariableop_60_adam_residual_3_bias_v
identity_62ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ	RestoreV2ЂRestoreV2_1ф"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*№!
valueц!Bу!=B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*
valueB=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesп
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesї
є:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*K
dtypesA
?2=	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_21_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_21_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_22_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_22_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_23_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_23_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_24_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_24_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_25_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_25_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_26_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_26_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_27_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_27_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Ђ
AssignVariableOp_14AssignVariableOp)assignvariableop_14_class_output_3_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15 
AssignVariableOp_15AssignVariableOp'assignvariableop_15_class_output_3_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16
AssignVariableOp_16AssignVariableOp%assignvariableop_16_residual_3_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17
AssignVariableOp_17AssignVariableOp#assignvariableop_17_residual_3_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0	*
_output_shapes
:2
Identity_18
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25Ѓ
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_21_kernel_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26Ё
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_21_bias_mIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27Ѓ
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_22_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28Ё
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_22_bias_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29Ѓ
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_23_kernel_mIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30Ё
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_23_bias_mIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31Ѓ
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_24_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32Ё
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_24_bias_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33Ѓ
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_25_kernel_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34Ё
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_25_bias_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35Ѓ
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_26_kernel_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36Ё
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_26_bias_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37Ѓ
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_27_kernel_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38Ё
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_27_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39Љ
AssignVariableOp_39AssignVariableOp0assignvariableop_39_adam_class_output_3_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40Ї
AssignVariableOp_40AssignVariableOp.assignvariableop_40_adam_class_output_3_bias_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41Ѕ
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adam_residual_3_kernel_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42Ѓ
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_residual_3_bias_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43Ѓ
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_21_kernel_vIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44Ё
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_21_bias_vIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45Ѓ
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_22_kernel_vIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46Ё
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_22_bias_vIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47Ѓ
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_23_kernel_vIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48Ё
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_23_bias_vIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49Ѓ
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_24_kernel_vIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50Ё
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_24_bias_vIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51Ѓ
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_25_kernel_vIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52Ё
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_25_bias_vIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53Ѓ
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_26_kernel_vIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54Ё
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_26_bias_vIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55Ѓ
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_27_kernel_vIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56Ё
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_27_bias_vIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57Љ
AssignVariableOp_57AssignVariableOp0assignvariableop_57_adam_class_output_3_kernel_vIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58Ї
AssignVariableOp_58AssignVariableOp.assignvariableop_58_adam_class_output_3_bias_vIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59Ѕ
AssignVariableOp_59AssignVariableOp,assignvariableop_59_adam_residual_3_kernel_vIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60Ѓ
AssignVariableOp_60AssignVariableOp*assignvariableop_60_adam_residual_3_bias_vIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60Ј
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesФ
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
NoOp
Identity_61Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_61Љ
Identity_62IdentityIdentity_61:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_62"#
identity_62Identity_62:output:0*
_input_shapesљ
і: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
ъ
ѕ
)__inference_model_3_layer_call_fn_2398178

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
identityЂStatefulPartitionedCallЌ
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
:џџџџџџџџџ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_23978612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
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

d
+__inference_dropout_1_layer_call_fn_2398300

inputs
identityЂStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_23975222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
я
­
E__inference_dense_24_layer_call_and_return_conditional_losses_2398249

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
џ

*__inference_dense_24_layer_call_fn_2398258

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_23974672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ь
­
E__inference_dense_23_layer_call_and_return_conditional_losses_2398229

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :::O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
щ
­
E__inference_dense_27_layer_call_and_return_conditional_losses_2397578

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ы7
З
D__inference_model_3_layer_call_and_return_conditional_losses_2397715
input_4
dense_21_2397667
dense_21_2397669
dense_22_2397672
dense_22_2397674
dense_23_2397677
dense_23_2397679
dense_24_2397682
dense_24_2397684
dense_25_2397687
dense_25_2397689
dense_26_2397693
dense_26_2397695
dense_27_2397698
dense_27_2397700
class_output_2397703
class_output_2397705
residual_2397708
residual_2397710
identityЂ$class_output/StatefulPartitionedCallЂ dense_21/StatefulPartitionedCallЂ dense_22/StatefulPartitionedCallЂ dense_23/StatefulPartitionedCallЂ dense_24/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ dense_26/StatefulPartitionedCallЂ dense_27/StatefulPartitionedCallЂ residual/StatefulPartitionedCallљ
 dense_21/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_21_2397667dense_21_2397669*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_23973862"
 dense_21/StatefulPartitionedCall
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_2397672dense_22_2397674*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_23974132"
 dense_22/StatefulPartitionedCall
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_2397677dense_23_2397679*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_23974402"
 dense_23/StatefulPartitionedCall
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_2397682dense_24_2397684*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_23974672"
 dense_24/StatefulPartitionedCall
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_2397687dense_25_2397689*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_23974942"
 dense_25/StatefulPartitionedCallн
dropout_1/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_23975272
dropout_1/PartitionedCall
 dense_26/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_26_2397693dense_26_2397695*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_23975512"
 dense_26/StatefulPartitionedCall
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_2397698dense_27_2397700*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_23975782"
 dense_27/StatefulPartitionedCallЏ
$class_output/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0class_output_2397703class_output_2397705*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_class_output_layer_call_and_return_conditional_losses_23976052&
$class_output/StatefulPartitionedCall
 residual/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0residual_2397708residual_2397710*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_residual_layer_call_and_return_conditional_losses_23976312"
 residual/StatefulPartitionedCall
concatenate_3/PartitionedCallPartitionedCall-class_output/StatefulPartitionedCall:output:0)residual/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_23976542
concatenate_3/PartitionedCallЙ
IdentityIdentity&concatenate_3/PartitionedCall:output:0%^class_output/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^residual/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ::::::::::::::::::2L
$class_output/StatefulPartitionedCall$class_output/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 residual/StatefulPartitionedCall residual/StatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_4:
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

­
E__inference_residual_layer_call_and_return_conditional_losses_2398375

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :::O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
9
л
D__inference_model_3_layer_call_and_return_conditional_losses_2397664
input_4
dense_21_2397397
dense_21_2397399
dense_22_2397424
dense_22_2397426
dense_23_2397451
dense_23_2397453
dense_24_2397478
dense_24_2397480
dense_25_2397505
dense_25_2397507
dense_26_2397562
dense_26_2397564
dense_27_2397589
dense_27_2397591
class_output_2397616
class_output_2397618
residual_2397642
residual_2397644
identityЂ$class_output/StatefulPartitionedCallЂ dense_21/StatefulPartitionedCallЂ dense_22/StatefulPartitionedCallЂ dense_23/StatefulPartitionedCallЂ dense_24/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ dense_26/StatefulPartitionedCallЂ dense_27/StatefulPartitionedCallЂ!dropout_1/StatefulPartitionedCallЂ residual/StatefulPartitionedCallљ
 dense_21/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_21_2397397dense_21_2397399*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_23973862"
 dense_21/StatefulPartitionedCall
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_2397424dense_22_2397426*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_23974132"
 dense_22/StatefulPartitionedCall
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_2397451dense_23_2397453*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_23974402"
 dense_23/StatefulPartitionedCall
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_2397478dense_24_2397480*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_23974672"
 dense_24/StatefulPartitionedCall
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_2397505dense_25_2397507*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_23974942"
 dense_25/StatefulPartitionedCallѕ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_23975222#
!dropout_1/StatefulPartitionedCall
 dense_26/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_26_2397562dense_26_2397564*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_23975512"
 dense_26/StatefulPartitionedCall
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_2397589dense_27_2397591*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_23975782"
 dense_27/StatefulPartitionedCallЏ
$class_output/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0class_output_2397616class_output_2397618*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_class_output_layer_call_and_return_conditional_losses_23976052&
$class_output/StatefulPartitionedCall
 residual/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0residual_2397642residual_2397644*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_residual_layer_call_and_return_conditional_losses_23976312"
 residual/StatefulPartitionedCall
concatenate_3/PartitionedCallPartitionedCall-class_output/StatefulPartitionedCall:output:0)residual/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_23976542
concatenate_3/PartitionedCallн
IdentityIdentity&concatenate_3/PartitionedCall:output:0%^class_output/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall!^residual/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ::::::::::::::::::2L
$class_output/StatefulPartitionedCall$class_output/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2D
 residual/StatefulPartitionedCall residual/StatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_4:
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
Ж
t
J__inference_concatenate_3_layer_call_and_return_conditional_losses_2397654

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
:џџџџџџџџџ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ
ѕ
)__inference_model_3_layer_call_fn_2398137

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
identityЂStatefulPartitionedCallЌ
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
:џџџџџџџџџ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_23977692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
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
9
к
D__inference_model_3_layer_call_and_return_conditional_losses_2397769

inputs
dense_21_2397721
dense_21_2397723
dense_22_2397726
dense_22_2397728
dense_23_2397731
dense_23_2397733
dense_24_2397736
dense_24_2397738
dense_25_2397741
dense_25_2397743
dense_26_2397747
dense_26_2397749
dense_27_2397752
dense_27_2397754
class_output_2397757
class_output_2397759
residual_2397762
residual_2397764
identityЂ$class_output/StatefulPartitionedCallЂ dense_21/StatefulPartitionedCallЂ dense_22/StatefulPartitionedCallЂ dense_23/StatefulPartitionedCallЂ dense_24/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ dense_26/StatefulPartitionedCallЂ dense_27/StatefulPartitionedCallЂ!dropout_1/StatefulPartitionedCallЂ residual/StatefulPartitionedCallј
 dense_21/StatefulPartitionedCallStatefulPartitionedCallinputsdense_21_2397721dense_21_2397723*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_23973862"
 dense_21/StatefulPartitionedCall
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_2397726dense_22_2397728*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_23974132"
 dense_22/StatefulPartitionedCall
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_2397731dense_23_2397733*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_23974402"
 dense_23/StatefulPartitionedCall
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_2397736dense_24_2397738*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_23974672"
 dense_24/StatefulPartitionedCall
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_2397741dense_25_2397743*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_23974942"
 dense_25/StatefulPartitionedCallѕ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_23975222#
!dropout_1/StatefulPartitionedCall
 dense_26/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_26_2397747dense_26_2397749*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_23975512"
 dense_26/StatefulPartitionedCall
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_2397752dense_27_2397754*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_23975782"
 dense_27/StatefulPartitionedCallЏ
$class_output/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0class_output_2397757class_output_2397759*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_class_output_layer_call_and_return_conditional_losses_23976052&
$class_output/StatefulPartitionedCall
 residual/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0residual_2397762residual_2397764*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_residual_layer_call_and_return_conditional_losses_23976312"
 residual/StatefulPartitionedCall
concatenate_3/PartitionedCallPartitionedCall-class_output/StatefulPartitionedCall:output:0)residual/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_23976542
concatenate_3/PartitionedCallн
IdentityIdentity&concatenate_3/PartitionedCall:output:0%^class_output/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall!^residual/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ::::::::::::::::::2L
$class_output/StatefulPartitionedCall$class_output/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2D
 residual/StatefulPartitionedCall residual/StatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
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

e
F__inference_dropout_1_layer_call_and_return_conditional_losses_2398290

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Э
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_2398295

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч
ђ
%__inference_signature_wrapper_2397951
input_4
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
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:џџџџџџџџџ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__wrapped_model_23973712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_4:
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

­
E__inference_residual_layer_call_and_return_conditional_losses_2397631

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :::O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
щ
­
E__inference_dense_21_layer_call_and_return_conditional_losses_2397386

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Э
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_2397527

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ь
­
E__inference_dense_23_layer_call_and_return_conditional_losses_2397440

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :::O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

e
F__inference_dropout_1_layer_call_and_return_conditional_losses_2397522

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
э
і
)__inference_model_3_layer_call_fn_2397808
input_4
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
identityЂStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:џџџџџџџџџ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_23977692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_4:
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
§

*__inference_dense_23_layer_call_fn_2398238

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_23974402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
I

D__inference_model_3_layer_call_and_return_conditional_losses_2398096

inputs+
'dense_21_matmul_readvariableop_resource,
(dense_21_biasadd_readvariableop_resource+
'dense_22_matmul_readvariableop_resource,
(dense_22_biasadd_readvariableop_resource+
'dense_23_matmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource+
'dense_24_matmul_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource+
'dense_25_matmul_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource/
+class_output_matmul_readvariableop_resource0
,class_output_biasadd_readvariableop_resource+
'residual_matmul_readvariableop_resource,
(residual_biasadd_readvariableop_resource
identityЉ
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_21/MatMul/ReadVariableOp
dense_21/MatMulMatMulinputs&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_21/MatMulЇ
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_21/BiasAdd/ReadVariableOpЅ
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_21/BiasAdds
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_21/ReluЈ
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
dense_22/MatMul/ReadVariableOpЃ
dense_22/MatMulMatMuldense_21/Relu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_22/MatMulЇ
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_22/BiasAdd/ReadVariableOpЅ
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_22/BiasAdds
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_22/ReluЉ
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02 
dense_23/MatMul/ReadVariableOpЄ
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_23/MatMulЈ
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpІ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_23/BiasAddt
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_23/ReluЊ
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_24/MatMul/ReadVariableOpЄ
dense_24/MatMulMatMuldense_23/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_24/MatMulЈ
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_24/BiasAdd/ReadVariableOpІ
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_24/BiasAddt
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_24/ReluЊ
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_25/MatMul/ReadVariableOpЄ
dense_25/MatMulMatMuldense_24/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_25/MatMulЈ
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_25/BiasAdd/ReadVariableOpІ
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_25/BiasAddt
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_25/Relu
dropout_1/IdentityIdentitydense_25/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_1/IdentityЊ
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_26/MatMul/ReadVariableOpЄ
dense_26/MatMulMatMuldropout_1/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_26/MatMulЈ
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_26/BiasAdd/ReadVariableOpІ
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_26/BiasAddt
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_26/ReluЉ
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02 
dense_27/MatMul/ReadVariableOpЃ
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_27/MatMulЇ
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_27/BiasAdd/ReadVariableOpЅ
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_27/BiasAdds
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_27/ReluД
"class_output/MatMul/ReadVariableOpReadVariableOp+class_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"class_output/MatMul/ReadVariableOpЏ
class_output/MatMulMatMuldense_27/Relu:activations:0*class_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
class_output/MatMulГ
#class_output/BiasAdd/ReadVariableOpReadVariableOp,class_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#class_output/BiasAdd/ReadVariableOpЕ
class_output/BiasAddBiasAddclass_output/MatMul:product:0+class_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
class_output/BiasAdd
class_output/SoftmaxSoftmaxclass_output/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
class_output/SoftmaxЈ
residual/MatMul/ReadVariableOpReadVariableOp'residual_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
residual/MatMul/ReadVariableOpЃ
residual/MatMulMatMuldense_27/Relu:activations:0&residual/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
residual/MatMulЇ
residual/BiasAdd/ReadVariableOpReadVariableOp(residual_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
residual/BiasAdd/ReadVariableOpЅ
residual/BiasAddBiasAddresidual/MatMul:product:0'residual/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
residual/BiasAddx
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axisв
concatenate_3/concatConcatV2class_output/Softmax:softmax:0residual/BiasAdd:output:0"concatenate_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2
concatenate_3/concatq
IdentityIdentityconcatenate_3/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ:::::::::::::::::::P L
(
_output_shapes
:џџџџџџџџџ
 
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
ћ

*__inference_dense_22_layer_call_fn_2398218

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_23974132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
џ

*__inference_dense_26_layer_call_fn_2398325

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_23975512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
§

*__inference_dense_27_layer_call_fn_2398345

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_23975782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
щ
­
E__inference_dense_21_layer_call_and_return_conditional_losses_2398189

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
я
­
E__inference_dense_25_layer_call_and_return_conditional_losses_2397494

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ц
­
E__inference_dense_22_layer_call_and_return_conditional_losses_2398209

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
э
і
)__inference_model_3_layer_call_fn_2397900
input_4
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
identityЂStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:џџџџџџџџџ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_23978612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_4:
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
ђ
Б
I__inference_class_output_layer_call_and_return_conditional_losses_2397605

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :::O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ш7
Ж
D__inference_model_3_layer_call_and_return_conditional_losses_2397861

inputs
dense_21_2397813
dense_21_2397815
dense_22_2397818
dense_22_2397820
dense_23_2397823
dense_23_2397825
dense_24_2397828
dense_24_2397830
dense_25_2397833
dense_25_2397835
dense_26_2397839
dense_26_2397841
dense_27_2397844
dense_27_2397846
class_output_2397849
class_output_2397851
residual_2397854
residual_2397856
identityЂ$class_output/StatefulPartitionedCallЂ dense_21/StatefulPartitionedCallЂ dense_22/StatefulPartitionedCallЂ dense_23/StatefulPartitionedCallЂ dense_24/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ dense_26/StatefulPartitionedCallЂ dense_27/StatefulPartitionedCallЂ residual/StatefulPartitionedCallј
 dense_21/StatefulPartitionedCallStatefulPartitionedCallinputsdense_21_2397813dense_21_2397815*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_23973862"
 dense_21/StatefulPartitionedCall
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_2397818dense_22_2397820*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_23974132"
 dense_22/StatefulPartitionedCall
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_2397823dense_23_2397825*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_23974402"
 dense_23/StatefulPartitionedCall
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_2397828dense_24_2397830*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_23974672"
 dense_24/StatefulPartitionedCall
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_2397833dense_25_2397835*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_23974942"
 dense_25/StatefulPartitionedCallн
dropout_1/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_23975272
dropout_1/PartitionedCall
 dense_26/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_26_2397839dense_26_2397841*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_23975512"
 dense_26/StatefulPartitionedCall
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_2397844dense_27_2397846*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_23975782"
 dense_27/StatefulPartitionedCallЏ
$class_output/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0class_output_2397849class_output_2397851*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_class_output_layer_call_and_return_conditional_losses_23976052&
$class_output/StatefulPartitionedCall
 residual/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0residual_2397854residual_2397856*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_residual_layer_call_and_return_conditional_losses_23976312"
 residual/StatefulPartitionedCall
concatenate_3/PartitionedCallPartitionedCall-class_output/StatefulPartitionedCall:output:0)residual/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_23976542
concatenate_3/PartitionedCallЙ
IdentityIdentity&concatenate_3/PartitionedCall:output:0%^class_output/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^residual/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ::::::::::::::::::2L
$class_output/StatefulPartitionedCall$class_output/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 residual/StatefulPartitionedCall residual/StatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
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
џ

*__inference_dense_25_layer_call_fn_2398278

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_23974942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
я
­
E__inference_dense_26_layer_call_and_return_conditional_losses_2398316

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
я
­
E__inference_dense_25_layer_call_and_return_conditional_losses_2398269

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ЧR

D__inference_model_3_layer_call_and_return_conditional_losses_2398027

inputs+
'dense_21_matmul_readvariableop_resource,
(dense_21_biasadd_readvariableop_resource+
'dense_22_matmul_readvariableop_resource,
(dense_22_biasadd_readvariableop_resource+
'dense_23_matmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource+
'dense_24_matmul_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource+
'dense_25_matmul_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource/
+class_output_matmul_readvariableop_resource0
,class_output_biasadd_readvariableop_resource+
'residual_matmul_readvariableop_resource,
(residual_biasadd_readvariableop_resource
identityЉ
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_21/MatMul/ReadVariableOp
dense_21/MatMulMatMulinputs&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_21/MatMulЇ
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_21/BiasAdd/ReadVariableOpЅ
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_21/BiasAdds
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_21/ReluЈ
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
dense_22/MatMul/ReadVariableOpЃ
dense_22/MatMulMatMuldense_21/Relu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_22/MatMulЇ
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_22/BiasAdd/ReadVariableOpЅ
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_22/BiasAdds
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_22/ReluЉ
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02 
dense_23/MatMul/ReadVariableOpЄ
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_23/MatMulЈ
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_23/BiasAdd/ReadVariableOpІ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_23/BiasAddt
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_23/ReluЊ
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_24/MatMul/ReadVariableOpЄ
dense_24/MatMulMatMuldense_23/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_24/MatMulЈ
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_24/BiasAdd/ReadVariableOpІ
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_24/BiasAddt
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_24/ReluЊ
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_25/MatMul/ReadVariableOpЄ
dense_25/MatMulMatMuldense_24/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_25/MatMulЈ
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_25/BiasAdd/ReadVariableOpІ
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_25/BiasAddt
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_25/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/ConstЇ
dropout_1/dropout/MulMuldense_25/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_1/dropout/Mul}
dropout_1/dropout/ShapeShapedense_25/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shapeг
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yч
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2 
dropout_1/dropout/GreaterEqual
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout_1/dropout/CastЃ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_1/dropout/Mul_1Њ
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_26/MatMul/ReadVariableOpЄ
dense_26/MatMulMatMuldropout_1/dropout/Mul_1:z:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_26/MatMulЈ
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_26/BiasAdd/ReadVariableOpІ
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_26/BiasAddt
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_26/ReluЉ
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02 
dense_27/MatMul/ReadVariableOpЃ
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_27/MatMulЇ
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_27/BiasAdd/ReadVariableOpЅ
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_27/BiasAdds
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_27/ReluД
"class_output/MatMul/ReadVariableOpReadVariableOp+class_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"class_output/MatMul/ReadVariableOpЏ
class_output/MatMulMatMuldense_27/Relu:activations:0*class_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
class_output/MatMulГ
#class_output/BiasAdd/ReadVariableOpReadVariableOp,class_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#class_output/BiasAdd/ReadVariableOpЕ
class_output/BiasAddBiasAddclass_output/MatMul:product:0+class_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
class_output/BiasAdd
class_output/SoftmaxSoftmaxclass_output/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
class_output/SoftmaxЈ
residual/MatMul/ReadVariableOpReadVariableOp'residual_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
residual/MatMul/ReadVariableOpЃ
residual/MatMulMatMuldense_27/Relu:activations:0&residual/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
residual/MatMulЇ
residual/BiasAdd/ReadVariableOpReadVariableOp(residual_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
residual/BiasAdd/ReadVariableOpЅ
residual/BiasAddBiasAddresidual/MatMul:product:0'residual/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
residual/BiasAddx
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axisв
concatenate_3/concatConcatV2class_output/Softmax:softmax:0residual/BiasAdd:output:0"concatenate_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2
concatenate_3/concatq
IdentityIdentityconcatenate_3/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ:::::::::::::::::::P L
(
_output_shapes
:џџџџџџџџџ
 
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
ц
­
E__inference_dense_22_layer_call_and_return_conditional_losses_2397413

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ђ
Б
I__inference_class_output_layer_call_and_return_conditional_losses_2398356

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :::O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Џ
Х
 __inference__traced_save_2398607
file_prefix.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop4
0savev2_class_output_3_kernel_read_readvariableop2
.savev2_class_output_3_bias_read_readvariableop0
,savev2_residual_3_kernel_read_readvariableop.
*savev2_residual_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_21_kernel_m_read_readvariableop3
/savev2_adam_dense_21_bias_m_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableop5
1savev2_adam_dense_24_kernel_m_read_readvariableop3
/savev2_adam_dense_24_bias_m_read_readvariableop5
1savev2_adam_dense_25_kernel_m_read_readvariableop3
/savev2_adam_dense_25_bias_m_read_readvariableop5
1savev2_adam_dense_26_kernel_m_read_readvariableop3
/savev2_adam_dense_26_bias_m_read_readvariableop5
1savev2_adam_dense_27_kernel_m_read_readvariableop3
/savev2_adam_dense_27_bias_m_read_readvariableop;
7savev2_adam_class_output_3_kernel_m_read_readvariableop9
5savev2_adam_class_output_3_bias_m_read_readvariableop7
3savev2_adam_residual_3_kernel_m_read_readvariableop5
1savev2_adam_residual_3_bias_m_read_readvariableop5
1savev2_adam_dense_21_kernel_v_read_readvariableop3
/savev2_adam_dense_21_bias_v_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableop5
1savev2_adam_dense_24_kernel_v_read_readvariableop3
/savev2_adam_dense_24_bias_v_read_readvariableop5
1savev2_adam_dense_25_kernel_v_read_readvariableop3
/savev2_adam_dense_25_bias_v_read_readvariableop5
1savev2_adam_dense_26_kernel_v_read_readvariableop3
/savev2_adam_dense_26_bias_v_read_readvariableop5
1savev2_adam_dense_27_kernel_v_read_readvariableop3
/savev2_adam_dense_27_bias_v_read_readvariableop;
7savev2_adam_class_output_3_kernel_v_read_readvariableop9
5savev2_adam_class_output_3_bias_v_read_readvariableop7
3savev2_adam_residual_3_kernel_v_read_readvariableop5
1savev2_adam_residual_3_bias_v_read_readvariableop
savev2_1_const

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_f0651155fc4b4243bae392daa3832fb4/part2	
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameо"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*№!
valueц!Bу!=B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*
valueB=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЖ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop0savev2_class_output_3_kernel_read_readvariableop.savev2_class_output_3_bias_read_readvariableop,savev2_residual_3_kernel_read_readvariableop*savev2_residual_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_21_kernel_m_read_readvariableop/savev2_adam_dense_21_bias_m_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableop1savev2_adam_dense_25_kernel_m_read_readvariableop/savev2_adam_dense_25_bias_m_read_readvariableop1savev2_adam_dense_26_kernel_m_read_readvariableop/savev2_adam_dense_26_bias_m_read_readvariableop1savev2_adam_dense_27_kernel_m_read_readvariableop/savev2_adam_dense_27_bias_m_read_readvariableop7savev2_adam_class_output_3_kernel_m_read_readvariableop5savev2_adam_class_output_3_bias_m_read_readvariableop3savev2_adam_residual_3_kernel_m_read_readvariableop1savev2_adam_residual_3_bias_m_read_readvariableop1savev2_adam_dense_21_kernel_v_read_readvariableop/savev2_adam_dense_21_bias_v_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableop1savev2_adam_dense_25_kernel_v_read_readvariableop/savev2_adam_dense_25_bias_v_read_readvariableop1savev2_adam_dense_26_kernel_v_read_readvariableop/savev2_adam_dense_26_bias_v_read_readvariableop1savev2_adam_dense_27_kernel_v_read_readvariableop/savev2_adam_dense_27_bias_v_read_readvariableop7savev2_adam_class_output_3_kernel_v_read_readvariableop5savev2_adam_class_output_3_bias_v_read_readvariableop3savev2_adam_residual_3_kernel_v_read_readvariableop1savev2_adam_residual_3_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *K
dtypesA
?2=	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardЌ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ђ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesЯ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЌ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ў
_input_shapesь
щ: :	@:@:@ : :	 ::
::
::
::	 : : :: :: : : : : : : :	@:@:@ : :	 ::
::
::
::	 : : :: ::	@:@:@ : :	 ::
::
::
::	 : : :: :: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :%!

_output_shapes
:	 :!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&	"
 
_output_shapes
:
:!


_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	 : 
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
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :%!

_output_shapes
:	 :!

_output_shapes	
::& "
 
_output_shapes
:
:!!

_output_shapes	
::&""
 
_output_shapes
:
:!#

_output_shapes	
::&$"
 
_output_shapes
:
:!%

_output_shapes	
::%&!

_output_shapes
:	 : '
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
:	@: -

_output_shapes
:@:$. 

_output_shapes

:@ : /

_output_shapes
: :%0!

_output_shapes
:	 :!1

_output_shapes	
::&2"
 
_output_shapes
:
:!3

_output_shapes	
::&4"
 
_output_shapes
:
:!5

_output_shapes	
::&6"
 
_output_shapes
:
:!7

_output_shapes	
::%8!

_output_shapes
:	 : 9
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
T

"__inference__wrapped_model_2397371
input_43
/model_3_dense_21_matmul_readvariableop_resource4
0model_3_dense_21_biasadd_readvariableop_resource3
/model_3_dense_22_matmul_readvariableop_resource4
0model_3_dense_22_biasadd_readvariableop_resource3
/model_3_dense_23_matmul_readvariableop_resource4
0model_3_dense_23_biasadd_readvariableop_resource3
/model_3_dense_24_matmul_readvariableop_resource4
0model_3_dense_24_biasadd_readvariableop_resource3
/model_3_dense_25_matmul_readvariableop_resource4
0model_3_dense_25_biasadd_readvariableop_resource3
/model_3_dense_26_matmul_readvariableop_resource4
0model_3_dense_26_biasadd_readvariableop_resource3
/model_3_dense_27_matmul_readvariableop_resource4
0model_3_dense_27_biasadd_readvariableop_resource7
3model_3_class_output_matmul_readvariableop_resource8
4model_3_class_output_biasadd_readvariableop_resource3
/model_3_residual_matmul_readvariableop_resource4
0model_3_residual_biasadd_readvariableop_resource
identityС
&model_3/dense_21/MatMul/ReadVariableOpReadVariableOp/model_3_dense_21_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&model_3/dense_21/MatMul/ReadVariableOpЇ
model_3/dense_21/MatMulMatMulinput_4.model_3/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
model_3/dense_21/MatMulП
'model_3/dense_21/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_3/dense_21/BiasAdd/ReadVariableOpХ
model_3/dense_21/BiasAddBiasAdd!model_3/dense_21/MatMul:product:0/model_3/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
model_3/dense_21/BiasAdd
model_3/dense_21/ReluRelu!model_3/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
model_3/dense_21/ReluР
&model_3/dense_22/MatMul/ReadVariableOpReadVariableOp/model_3_dense_22_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02(
&model_3/dense_22/MatMul/ReadVariableOpУ
model_3/dense_22/MatMulMatMul#model_3/dense_21/Relu:activations:0.model_3/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
model_3/dense_22/MatMulП
'model_3/dense_22/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_3/dense_22/BiasAdd/ReadVariableOpХ
model_3/dense_22/BiasAddBiasAdd!model_3/dense_22/MatMul:product:0/model_3/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
model_3/dense_22/BiasAdd
model_3/dense_22/ReluRelu!model_3/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
model_3/dense_22/ReluС
&model_3/dense_23/MatMul/ReadVariableOpReadVariableOp/model_3_dense_23_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02(
&model_3/dense_23/MatMul/ReadVariableOpФ
model_3/dense_23/MatMulMatMul#model_3/dense_22/Relu:activations:0.model_3/dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_3/dense_23/MatMulР
'model_3/dense_23/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'model_3/dense_23/BiasAdd/ReadVariableOpЦ
model_3/dense_23/BiasAddBiasAdd!model_3/dense_23/MatMul:product:0/model_3/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_3/dense_23/BiasAdd
model_3/dense_23/ReluRelu!model_3/dense_23/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_3/dense_23/ReluТ
&model_3/dense_24/MatMul/ReadVariableOpReadVariableOp/model_3_dense_24_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02(
&model_3/dense_24/MatMul/ReadVariableOpФ
model_3/dense_24/MatMulMatMul#model_3/dense_23/Relu:activations:0.model_3/dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_3/dense_24/MatMulР
'model_3/dense_24/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'model_3/dense_24/BiasAdd/ReadVariableOpЦ
model_3/dense_24/BiasAddBiasAdd!model_3/dense_24/MatMul:product:0/model_3/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_3/dense_24/BiasAdd
model_3/dense_24/ReluRelu!model_3/dense_24/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_3/dense_24/ReluТ
&model_3/dense_25/MatMul/ReadVariableOpReadVariableOp/model_3_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02(
&model_3/dense_25/MatMul/ReadVariableOpФ
model_3/dense_25/MatMulMatMul#model_3/dense_24/Relu:activations:0.model_3/dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_3/dense_25/MatMulР
'model_3/dense_25/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'model_3/dense_25/BiasAdd/ReadVariableOpЦ
model_3/dense_25/BiasAddBiasAdd!model_3/dense_25/MatMul:product:0/model_3/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_3/dense_25/BiasAdd
model_3/dense_25/ReluRelu!model_3/dense_25/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_3/dense_25/Relu
model_3/dropout_1/IdentityIdentity#model_3/dense_25/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_3/dropout_1/IdentityТ
&model_3/dense_26/MatMul/ReadVariableOpReadVariableOp/model_3_dense_26_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02(
&model_3/dense_26/MatMul/ReadVariableOpФ
model_3/dense_26/MatMulMatMul#model_3/dropout_1/Identity:output:0.model_3/dense_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_3/dense_26/MatMulР
'model_3/dense_26/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_26_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'model_3/dense_26/BiasAdd/ReadVariableOpЦ
model_3/dense_26/BiasAddBiasAdd!model_3/dense_26/MatMul:product:0/model_3/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_3/dense_26/BiasAdd
model_3/dense_26/ReluRelu!model_3/dense_26/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_3/dense_26/ReluС
&model_3/dense_27/MatMul/ReadVariableOpReadVariableOp/model_3_dense_27_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02(
&model_3/dense_27/MatMul/ReadVariableOpУ
model_3/dense_27/MatMulMatMul#model_3/dense_26/Relu:activations:0.model_3/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
model_3/dense_27/MatMulП
'model_3/dense_27/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_3/dense_27/BiasAdd/ReadVariableOpХ
model_3/dense_27/BiasAddBiasAdd!model_3/dense_27/MatMul:product:0/model_3/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
model_3/dense_27/BiasAdd
model_3/dense_27/ReluRelu!model_3/dense_27/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
model_3/dense_27/ReluЬ
*model_3/class_output/MatMul/ReadVariableOpReadVariableOp3model_3_class_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*model_3/class_output/MatMul/ReadVariableOpЯ
model_3/class_output/MatMulMatMul#model_3/dense_27/Relu:activations:02model_3/class_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_3/class_output/MatMulЫ
+model_3/class_output/BiasAdd/ReadVariableOpReadVariableOp4model_3_class_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_3/class_output/BiasAdd/ReadVariableOpе
model_3/class_output/BiasAddBiasAdd%model_3/class_output/MatMul:product:03model_3/class_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_3/class_output/BiasAdd 
model_3/class_output/SoftmaxSoftmax%model_3/class_output/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_3/class_output/SoftmaxР
&model_3/residual/MatMul/ReadVariableOpReadVariableOp/model_3_residual_matmul_readvariableop_resource*
_output_shapes

: *
dtype02(
&model_3/residual/MatMul/ReadVariableOpУ
model_3/residual/MatMulMatMul#model_3/dense_27/Relu:activations:0.model_3/residual/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_3/residual/MatMulП
'model_3/residual/BiasAdd/ReadVariableOpReadVariableOp0model_3_residual_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_3/residual/BiasAdd/ReadVariableOpХ
model_3/residual/BiasAddBiasAdd!model_3/residual/MatMul:product:0/model_3/residual/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_3/residual/BiasAdd
!model_3/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_3/concatenate_3/concat/axisњ
model_3/concatenate_3/concatConcatV2&model_3/class_output/Softmax:softmax:0!model_3/residual/BiasAdd:output:0*model_3/concatenate_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2
model_3/concatenate_3/concaty
IdentityIdentity%model_3/concatenate_3/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ:::::::::::::::::::Q M
(
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_4:
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
П
v
J__inference_concatenate_3_layer_call_and_return_conditional_losses_2398391
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
ћ

*__inference_residual_layer_call_fn_2398384

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_residual_layer_call_and_return_conditional_losses_23976312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

[
/__inference_concatenate_3_layer_call_fn_2398397
inputs_0
inputs_1
identityЖ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_23976542
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
щ
­
E__inference_dense_27_layer_call_and_return_conditional_losses_2398336

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
§

*__inference_dense_21_layer_call_fn_2398198

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_23973862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 


.__inference_class_output_layer_call_fn_2398365

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_class_output_layer_call_and_return_conditional_losses_23976052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: "ЏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Б
serving_default
<
input_41
serving_default_input_4:0џџџџџџџџџA
concatenate_30
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ый
г[
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
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer-11
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
Л_default_save_signature
М__call__
+Н&call_and_return_all_conditional_losses"§V
_tf_keras_modelуV{"class_name": "Model", "name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 148]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_21", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_22", "inbound_nodes": [[["dense_21", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_23", "inbound_nodes": [[["dense_22", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_24", "inbound_nodes": [[["dense_23", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_25", "inbound_nodes": [[["dense_24", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["dense_25", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_26", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_27", "inbound_nodes": [[["dense_26", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "class_output", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "class_output", "inbound_nodes": [[["dense_27", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "residual", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "residual", "inbound_nodes": [[["dense_27", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_3", "inbound_nodes": [[["class_output", 0, 0, {}], ["residual", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["concatenate_3", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 148]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 148]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_21", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_22", "inbound_nodes": [[["dense_21", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_23", "inbound_nodes": [[["dense_22", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_24", "inbound_nodes": [[["dense_23", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_25", "inbound_nodes": [[["dense_24", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["dense_25", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_26", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_27", "inbound_nodes": [[["dense_26", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "class_output", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "class_output", "inbound_nodes": [[["dense_27", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "residual", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "residual", "inbound_nodes": [[["dense_27", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_3", "inbound_nodes": [[["class_output", 0, 0, {}], ["residual", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["concatenate_3", 0, 0]]}}, "training_config": {"loss": "custom_mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
э"ъ
_tf_keras_input_layerЪ{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 148]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 148]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
г

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
О__call__
+П&call_and_return_all_conditional_losses"Ќ
_tf_keras_layer{"class_name": "Dense", "name": "dense_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 148}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 148]}}
б

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"Њ
_tf_keras_layer{"class_name": "Dense", "name": "dense_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
в

kernel
 bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
Т__call__
+У&call_and_return_all_conditional_losses"Ћ
_tf_keras_layer{"class_name": "Dense", "name": "dense_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
е

%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"Ў
_tf_keras_layer{"class_name": "Dense", "name": "dense_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
з

+kernel
,bias
-trainable_variables
.regularization_losses
/	variables
0	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses"А
_tf_keras_layer{"class_name": "Dense", "name": "dense_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
Ф
1trainable_variables
2regularization_losses
3	variables
4	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses"Г
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
ж

5kernel
6bias
7trainable_variables
8regularization_losses
9	variables
:	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses"Џ
_tf_keras_layer{"class_name": "Dense", "name": "dense_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
г

;kernel
<bias
=trainable_variables
>regularization_losses
?	variables
@	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses"Ќ
_tf_keras_layer{"class_name": "Dense", "name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
л

Akernel
Bbias
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
Ю__call__
+Я&call_and_return_all_conditional_losses"Д
_tf_keras_layer{"class_name": "Dense", "name": "class_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "class_output", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
в

Gkernel
Hbias
Itrainable_variables
Jregularization_losses
K	variables
L	keras_api
а__call__
+б&call_and_return_all_conditional_losses"Ћ
_tf_keras_layer{"class_name": "Dense", "name": "residual", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "residual", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
Њ
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
в__call__
+г&call_and_return_all_conditional_losses"
_tf_keras_layerџ{"class_name": "Concatenate", "name": "concatenate_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 3]}, {"class_name": "TensorShape", "items": [null, 3]}]}
Л
Qiter

Rbeta_1

Sbeta_2
	Tdecay
Ulearning_ratemmmmm m%m&m+m,m 5mЁ6mЂ;mЃ<mЄAmЅBmІGmЇHmЈvЉvЊvЋvЌv­ vЎ%vЏ&vА+vБ,vВ5vГ6vД;vЕ<vЖAvЗBvИGvЙHvК"
	optimizer
І
0
1
2
3
4
 5
%6
&7
+8
,9
510
611
;12
<13
A14
B15
G16
H17"
trackable_list_wrapper
 "
trackable_list_wrapper
І
0
1
2
3
4
 5
%6
&7
+8
,9
510
611
;12
<13
A14
B15
G16
H17"
trackable_list_wrapper
Ю
trainable_variables
Vmetrics
Wnon_trainable_variables

Xlayers
regularization_losses
Ylayer_regularization_losses
	variables
Zlayer_metrics
М__call__
Л_default_save_signature
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
-
дserving_default"
signature_map
": 	@2dense_21/kernel
:@2dense_21/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А
trainable_variables
[metrics
\non_trainable_variables

]layers
regularization_losses
^layer_regularization_losses
	variables
_layer_metrics
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
!:@ 2dense_22/kernel
: 2dense_22/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А
trainable_variables
`metrics
anon_trainable_variables

blayers
regularization_losses
clayer_regularization_losses
	variables
dlayer_metrics
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
": 	 2dense_23/kernel
:2dense_23/bias
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
А
!trainable_variables
emetrics
fnon_trainable_variables

glayers
"regularization_losses
hlayer_regularization_losses
#	variables
ilayer_metrics
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_24/kernel
:2dense_24/bias
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
А
'trainable_variables
jmetrics
knon_trainable_variables

llayers
(regularization_losses
mlayer_regularization_losses
)	variables
nlayer_metrics
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_25/kernel
:2dense_25/bias
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
А
-trainable_variables
ometrics
pnon_trainable_variables

qlayers
.regularization_losses
rlayer_regularization_losses
/	variables
slayer_metrics
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
1trainable_variables
tmetrics
unon_trainable_variables

vlayers
2regularization_losses
wlayer_regularization_losses
3	variables
xlayer_metrics
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_26/kernel
:2dense_26/bias
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
А
7trainable_variables
ymetrics
znon_trainable_variables

{layers
8regularization_losses
|layer_regularization_losses
9	variables
}layer_metrics
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
": 	 2dense_27/kernel
: 2dense_27/bias
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
Г
=trainable_variables
~metrics
non_trainable_variables
layers
>regularization_losses
 layer_regularization_losses
?	variables
layer_metrics
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
':% 2class_output_3/kernel
!:2class_output_3/bias
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
Е
Ctrainable_variables
metrics
non_trainable_variables
layers
Dregularization_losses
 layer_regularization_losses
E	variables
layer_metrics
Ю__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
#:! 2residual_3/kernel
:2residual_3/bias
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
Е
Itrainable_variables
metrics
non_trainable_variables
layers
Jregularization_losses
 layer_regularization_losses
K	variables
layer_metrics
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Mtrainable_variables
metrics
non_trainable_variables
layers
Nregularization_losses
 layer_regularization_losses
O	variables
layer_metrics
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
v
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
11"
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
П

total

count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
':%	@2Adam/dense_21/kernel/m
 :@2Adam/dense_21/bias/m
&:$@ 2Adam/dense_22/kernel/m
 : 2Adam/dense_22/bias/m
':%	 2Adam/dense_23/kernel/m
!:2Adam/dense_23/bias/m
(:&
2Adam/dense_24/kernel/m
!:2Adam/dense_24/bias/m
(:&
2Adam/dense_25/kernel/m
!:2Adam/dense_25/bias/m
(:&
2Adam/dense_26/kernel/m
!:2Adam/dense_26/bias/m
':%	 2Adam/dense_27/kernel/m
 : 2Adam/dense_27/bias/m
,:* 2Adam/class_output_3/kernel/m
&:$2Adam/class_output_3/bias/m
(:& 2Adam/residual_3/kernel/m
": 2Adam/residual_3/bias/m
':%	@2Adam/dense_21/kernel/v
 :@2Adam/dense_21/bias/v
&:$@ 2Adam/dense_22/kernel/v
 : 2Adam/dense_22/bias/v
':%	 2Adam/dense_23/kernel/v
!:2Adam/dense_23/bias/v
(:&
2Adam/dense_24/kernel/v
!:2Adam/dense_24/bias/v
(:&
2Adam/dense_25/kernel/v
!:2Adam/dense_25/bias/v
(:&
2Adam/dense_26/kernel/v
!:2Adam/dense_26/bias/v
':%	 2Adam/dense_27/kernel/v
 : 2Adam/dense_27/bias/v
,:* 2Adam/class_output_3/kernel/v
&:$2Adam/class_output_3/bias/v
(:& 2Adam/residual_3/kernel/v
": 2Adam/residual_3/bias/v
с2о
"__inference__wrapped_model_2397371З
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *'Ђ$
"
input_4џџџџџџџџџ
ђ2я
)__inference_model_3_layer_call_fn_2397808
)__inference_model_3_layer_call_fn_2398178
)__inference_model_3_layer_call_fn_2398137
)__inference_model_3_layer_call_fn_2397900Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
о2л
D__inference_model_3_layer_call_and_return_conditional_losses_2398096
D__inference_model_3_layer_call_and_return_conditional_losses_2397715
D__inference_model_3_layer_call_and_return_conditional_losses_2397664
D__inference_model_3_layer_call_and_return_conditional_losses_2398027Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
д2б
*__inference_dense_21_layer_call_fn_2398198Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_21_layer_call_and_return_conditional_losses_2398189Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_22_layer_call_fn_2398218Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_22_layer_call_and_return_conditional_losses_2398209Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_23_layer_call_fn_2398238Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_23_layer_call_and_return_conditional_losses_2398229Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_24_layer_call_fn_2398258Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_24_layer_call_and_return_conditional_losses_2398249Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_25_layer_call_fn_2398278Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_25_layer_call_and_return_conditional_losses_2398269Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
+__inference_dropout_1_layer_call_fn_2398305
+__inference_dropout_1_layer_call_fn_2398300Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ъ2Ч
F__inference_dropout_1_layer_call_and_return_conditional_losses_2398295
F__inference_dropout_1_layer_call_and_return_conditional_losses_2398290Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
д2б
*__inference_dense_26_layer_call_fn_2398325Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_26_layer_call_and_return_conditional_losses_2398316Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_27_layer_call_fn_2398345Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_27_layer_call_and_return_conditional_losses_2398336Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
и2е
.__inference_class_output_layer_call_fn_2398365Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_class_output_layer_call_and_return_conditional_losses_2398356Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_residual_layer_call_fn_2398384Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_residual_layer_call_and_return_conditional_losses_2398375Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
й2ж
/__inference_concatenate_3_layer_call_fn_2398397Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
є2ё
J__inference_concatenate_3_layer_call_and_return_conditional_losses_2398391Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
4B2
%__inference_signature_wrapper_2397951input_4­
"__inference__wrapped_model_2397371 %&+,56;<ABGH1Ђ.
'Ђ$
"
input_4џџџџџџџџџ
Њ "=Њ:
8
concatenate_3'$
concatenate_3џџџџџџџџџЉ
I__inference_class_output_layer_call_and_return_conditional_losses_2398356\AB/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 
.__inference_class_output_layer_call_fn_2398365OAB/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџв
J__inference_concatenate_3_layer_call_and_return_conditional_losses_2398391ZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 Љ
/__inference_concatenate_3_layer_call_fn_2398397vZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
Њ "џџџџџџџџџІ
E__inference_dense_21_layer_call_and_return_conditional_losses_2398189]0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ@
 ~
*__inference_dense_21_layer_call_fn_2398198P0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ@Ѕ
E__inference_dense_22_layer_call_and_return_conditional_losses_2398209\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ 
 }
*__inference_dense_22_layer_call_fn_2398218O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ І
E__inference_dense_23_layer_call_and_return_conditional_losses_2398229] /Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "&Ђ#

0џџџџџџџџџ
 ~
*__inference_dense_23_layer_call_fn_2398238P /Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџЇ
E__inference_dense_24_layer_call_and_return_conditional_losses_2398249^%&0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_dense_24_layer_call_fn_2398258Q%&0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЇ
E__inference_dense_25_layer_call_and_return_conditional_losses_2398269^+,0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_dense_25_layer_call_fn_2398278Q+,0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЇ
E__inference_dense_26_layer_call_and_return_conditional_losses_2398316^560Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_dense_26_layer_call_fn_2398325Q560Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
E__inference_dense_27_layer_call_and_return_conditional_losses_2398336];<0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ 
 ~
*__inference_dense_27_layer_call_fn_2398345P;<0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ Ј
F__inference_dropout_1_layer_call_and_return_conditional_losses_2398290^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "&Ђ#

0џџџџџџџџџ
 Ј
F__inference_dropout_1_layer_call_and_return_conditional_losses_2398295^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "&Ђ#

0џџџџџџџџџ
 
+__inference_dropout_1_layer_call_fn_2398300Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ
+__inference_dropout_1_layer_call_fn_2398305Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџО
D__inference_model_3_layer_call_and_return_conditional_losses_2397664v %&+,56;<ABGH9Ђ6
/Ђ,
"
input_4џџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 О
D__inference_model_3_layer_call_and_return_conditional_losses_2397715v %&+,56;<ABGH9Ђ6
/Ђ,
"
input_4џџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Н
D__inference_model_3_layer_call_and_return_conditional_losses_2398027u %&+,56;<ABGH8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Н
D__inference_model_3_layer_call_and_return_conditional_losses_2398096u %&+,56;<ABGH8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
)__inference_model_3_layer_call_fn_2397808i %&+,56;<ABGH9Ђ6
/Ђ,
"
input_4џџџџџџџџџ
p

 
Њ "џџџџџџџџџ
)__inference_model_3_layer_call_fn_2397900i %&+,56;<ABGH9Ђ6
/Ђ,
"
input_4џџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
)__inference_model_3_layer_call_fn_2398137h %&+,56;<ABGH8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
)__inference_model_3_layer_call_fn_2398178h %&+,56;<ABGH8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџЅ
E__inference_residual_layer_call_and_return_conditional_losses_2398375\GH/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_residual_layer_call_fn_2398384OGH/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџЛ
%__inference_signature_wrapper_2397951 %&+,56;<ABGH<Ђ9
Ђ 
2Њ/
-
input_4"
input_4џџџџџџџџџ"=Њ:
8
concatenate_3'$
concatenate_3џџџџџџџџџ