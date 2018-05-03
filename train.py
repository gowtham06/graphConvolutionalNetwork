from __future__ import print_function
from keras.layers import Input,Dropout,Flatten, Dense,Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt
from graph import GCN
from utils import *

import time

DATASET='citeseer'
FILTER='chebyshev'
MAX_DEGREE=2
SYM_NORM=True
NUM_EPOCHS=100
NUM_CLASSES=7

X,A,y=load_data(dataset=DATASET)
idx_train,idx_val,idx_test,y_train,y_val,y_test,train_mask=get_splits(y)

X/=X.sum(1).reshape(-1,1)

if FILTER=='localpool':
	# Using local pooling after graph convolution
	A_=preprocess_adj(A,SYM_NORM)
	support=1 #should atleast be one

	graph=[X,A_]
	G=[Input(shape=(None,None),batch_shape=(None,None),sparse=True)]

elif FILTER=='chebyshev':
	#using Chebyshev basis filters
	L=normalized_laplacian(A,SYM_NORM)
	L_scaled=rescale_laplacian(L)
	T_k=chebyshev_polynomial(L_scaled,MAX_DEGREE)
	support=MAX_DEGREE+1
	graph=[X]+T_k
	G=[Input(shape=(None,None),batch_shape=(None,None),sparse=True) for _ in range(support)]

else:
	raise Exception('Invalid filter type')

X_input=Input(shape=(X.shape[1],))

# Defining the model.
H=Dropout(0.3)(X_input)
H=GCN(25,support,activation='relu',kernel_regularizer=l2(5e-4))([H]+G)
H=Dropout(0.4)(H)
H=GCN(20,support,activation='relu',kernel_regularizer=l2(5e-4))([H]+G)
dropout_layer=Dropout(0.4)(H)
dense_layer=Dense(NUM_CLASSES)(dropout_layer)
Y=Activation('softmax')(dense_layer)
#Compiling the model
model=Model(inputs=[X_input]+G,outputs=Y)
best_val_loss=1.0
VALIDATION_ACC=[]
TRAINING_ACC=[]
TRAINING_LOSS=[]
VALDIATION_LOSS=[]
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.01))
for i in range(1,NUM_EPOCHS+1):
	t=time.time()

	model.fit(graph,y_train,sample_weight=train_mask,batch_size=A.shape[0],epochs=1,shuffle=False,verbose=0)
	preds=model.predict(graph,batch_size=A.shape[0])

	train_val_loss,train_val_acc=evaluate_prediction(preds,[y_train,y_val],[idx_train,idx_val])

	print("Epoch :{:04d}".format(i),
		  "train_loss={:.4f}".format(train_val_loss[0]),
		  "train_acc={:.4f}".format(train_val_acc[0]),
		  "val_loss={:.4f}".format(train_val_loss[1]),
		  "val_acc={:.4f}".format(train_val_acc[1]),
		  "time={:.4f}".format(time.time()-t))
	VALIDATION_ACC.append(train_val_acc[1])
	TRAINING_ACC.append(train_val_acc[0])
	VALDIATION_LOSS.append(train_val_loss[1])
	TRAINING_LOSS.append(train_val_loss[0])
	if train_val_loss[1]<best_val_loss:
		best_val_loss=train_val_loss[1]

test_loss,test_acc=evaluate_prediction(preds,[y_test],[idx_test])
print("Test set results:","loss={:.4f}".format(test_loss[0]),"accuracy={:.4f}".format(test_acc[0]))


#Plotting the results
plt.subplot(211)
plt.title("Accuracy")
plt.plot(TRAINING_ACC, color="g", label="Train")
plt.plot(VALIDATION_ACC, color="b", label="Validation")
plt.legend(loc="best")
plt.subplot(212)
plt.title("Loss")
plt.plot(TRAINING_LOSS, color="g", label="Train")
plt.plot(VALDIATION_LOSS, color="b", label="Validation")
plt.legend(loc="best")
plt.tight_layout()
# plt.show()
plt.savefig('plot_1.png')


