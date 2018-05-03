import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh,ArpackNoConvergence

def encode_one_hot(labels):
	classes=set(labels)
	classes_dict={c:np.identity(len(classes))[i,:] for i,c, in enumerate(classes)}
	labels_output=np.array(list(map(classes_dict.get,labels)),dtype=np.int32)
	return labels_output

def load_data(path="Data/CiteSeer/",dataset="cora"):
	idx_features_labels=np.genfromtxt("{}{}.content".format(path,dataset),dtype=np.dtype(str))
	features=sparse.csr_matrix(idx_features_labels[:,1:-1],dtype=np.float32)
	labels=encode_one_hot(idx_features_labels[:,-1])

	#Graph
	idx=np.array(idx_features_labels[:,0],dtype=np.int32)
	idx_map={j:i for i,j in enumerate(idx)}
	edges_unordered=np.genfromtxt("{}{}.cites".format(path,dataset),dtype=np.int32)
	edges=np.array(list(map(idx_map.get,edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
	adj=sparse.coo_matrix((np.ones(edges.shape[0]),(edges[:,0],edges[:,1])),
						  shape=(labels.shape[0],labels.shape[0]),dtype=np.float32)

	adj=adj+adj.T.multiply(adj.T>adj)-adj.multiply(adj.T>adj)

	print('Dataset has {} nodes,{} edges, {} features'.format(adj.shape[0],edges.shape[0],features.shape[1]))
	return features.todense(),adj,labels

def normalize_adj(adj,symmetric_measure=True):
	if symmetric_measure:
		d=sparse.diags(np.power(np.array(adj.sum(1)),-0.5).flatten(),0)
		a_norm=adj.dot(d).transpose().dot(d).tocsr()
	else:
		d=sparse.diags(np.power(np.array(adj.sum(1)),-1).flatten(),0)
		a_norm=d.dot(adj).tocsr()
	return a_norm

def preprocess_adj(adj,symmetric_measure=True):
	adj=adj+sparse.eye(adj.shape[0])
	adj=normalize_adj(adj,symmetric_measure)
	return adj

def sample_mask(idx,l):
	mask=np.zeros(l)
	mask[idx]=1
	return np.array(mask,dtype=np.bool)

def get_splits(y):
	idx_train=range(140)
	idx_val=range(200,500)
	idx_test=range(500,1500)
	y_train=np.zeros(y.shape,dtype=np.int32)
	y_val=np.zeros(y.shape,dtype=np.int32)
	y_test=np.zeros(y.shape,dtype=np.int32)
	y_train[idx_train]=y[idx_train]
	y_val[idx_val]=y[idx_val]
	y_test[idx_test]=y[idx_test]
	train_mask=sample_mask(idx_train,y.shape[0])
	return idx_train,idx_val,idx_test,y_train,y_val,y_test,train_mask

def cross_entropy(preds,labels):
	#returns catgorical cross entropy loss
	return np.mean(-1*np.log(np.extract(labels,preds)))

def accuracy(preds,labels):
	return np.mean(np.equal(np.argmax(labels,1),np.argmax(preds,1)))

def evaluate_prediction(preds,labels,indices):
	split_loss=[]
	split_acc=[]
	for y_split,index_split in zip(labels,indices):
		split_loss.append(cross_entropy(preds[index_split],y_split[index_split]))
		split_acc.append(accuracy(preds[index_split],y_split[index_split]))
	return split_loss,split_acc

def normalized_laplacian(adj,symmetric_measure=True):
	normalized_adj=normalize_adj(adj,symmetric_measure)
	laplacian=sparse.eye(adj.shape[0])-normalized_adj
	return laplacian

def rescale_laplacian(laplacian):
	try:
		largest_eigen_value=eigsh(laplacian,1,which='LM',return_eigenvectors=False)[0]
	except ArpackNoConvergence:
		largest_eigen_value=2 # default value if eigen value calculation does not converge

	rescaled_laplacian=(2./largest_eigen_value)*laplacian-sparse.eye(laplacian.shape[0])
	return rescaled_laplacian

def chebyshev_polynomial(X,k):
	T_k=[]
	T_k.append(sparse.eye(X.shape[0]).tocsr())
	T_k.append(X)

	def  chebyshev_recurrence(T_k_minus_one,T_k_minus_two):
		X_=sparse.csr_matrix(X,copy=True)
		return 2*X_.dot(T_k_minus_two)-T_k_minus_two	

	for i in range(2,k+1):
		T_k.append(chebyshev_recurrence(T_k[-1],T_k[-2]))
	return T_k


def sparse_to_tuple(sparse_mx):
	if not sparse.isspmatrix_coo(sparse_mx):
		sparse_mx=sparse_mx.tocoo()
	coords=np,vstack((sparse_mx.row,sparse_mx.col)).transpose()
	values=sparse_mx.data
	shape=sparse_mx.shape
	return coords,values,shape
