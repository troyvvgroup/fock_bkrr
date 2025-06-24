import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from pyscf import gto, scf, ao2mo, dft
import h5py
import math
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

def main():
	start=time.time()
	#the general goal here is to train a machine learning model to predict converged Fock matrices from SAD guesses for a specific system
	#our training set consists of some number SAD/Fock pairs from an AIMD run
	#these Fock matrices are currently in the AO basis
	#our ML method is a manual implementation of a kernel ridge regression
		#the chosen kernel (inner product) can be varied
	#This code assumes the following file structure:
	#	three subdirectories, one titled "converged_fock" and one titled "sad_guess" and one titled "xyz_coords"
	#	files in the converged_fock subdirectory are formatted "fock_TIMESTEP.npy" (1-indexed)
	#	files in the sad_guess subdirectory are formatted "sad_TIMESTEP.npy" (1-indexed)
	#	files in the xyz_coords subdirectory are formatted "TIMESTEP.xyz" (1-indexed)
	#the code currently assumes the molecule or system of interest is charge neutral with spin 1 when computing energies from fock matrices
	#The script can write the ML generated Fock matrices for various training sizes to a specified directory for further analysis
	#	writes each train size to a different .npy in a directory named manually
	#	But it is also capable of doing said analysis internally

	dir_name=sys.argv[1] #the name of the directory to write the predicted fock matrices to 
	alpha = float(sys.argv[2]) #regularization strength
	n_split=int(sys.argv[3]) #number of blocks to split into. Must evenly divide UT matrix
	#not clear to me yet what the constraints on this are, but for now I'm throwing an exception if it doesn't evenly divide the upper-triangular part of the matrix
	cutoff_t = int(sys.argv[4]) #for using a subset of the available time files. Either the number you wish to include (first N) or 0 to use all
	span_c = float(sys.argv[5]) #whether or not to use the span correction term. 0 for no, 1 for default, otherwise the scaling factor beta
	next_t = int(sys.argv[6]) #the number of future timesteps to predict. If greater than the number available, just cuts off

	subsampling = 0 #if != 0, the fraction of training points to subsample (rounded down)	
	if len(sys.argv) > 7:
		subsampling = float(sys.argv[7])

	#user parameters
	train_sizes=[50,100,250,500, 750, 1000, 1500, 2000]
	solve=False #whether or not to solve the generated Hamiltonians to find total energies, MO energies, etc. Options are all, False, or a specific train size
	save=False #whether or not to save the generated Hamiltonians		
	inner_product_type='classic' #only supports the Frobenius inner product right now (np.vdot). 
	scaling=False #standard scaling, applied to both inputs and outputs
	split_alpha = False #when False, alpha is applied to both the linear prediction and the span correction. When set to a value, used the set value for the alpha in the span correction
	diagonal_correct = False #implement averaging across the diagonal to correct asymmetry
	full_matrix_mae = True #If true, prints the test MAE for the full matrix only. Currently does not print anything else no matter what
	use_lt = True #whether or not to include lower triangular blocks
	deep_dive = False #whether or not to print detailed block-by-block error

	#this implementation only supports the blocked KRR splitting approach, but retains the functions necessary for Akimov's approach
	#this implementation also currently only implements the full cross-block approach

	#first, read in the matrices
	_, _, converged_files=next(os.walk("converged_fock"))
	_, _, sad_files=next(os.walk("sad_guess"))
	_, _, xyz_files=next(os.walk("xyz_coords"))
	if len(converged_files)!=len(sad_files): #or len(converged_files)!=len(xyz_files):
		raise Exception("Need equal # of sad/converged/xyz points")

	#sort them so they are all numbered 1-N timesteps
	converged_files=np.sort(converged_files)
	sad_files=np.sort(sad_files)
	xyz_files=np.sort(xyz_files)
	key=[int(fock[5:-4]) for fock in converged_files]
	sort_key=np.argsort(key)
	converged_files=np.take_along_axis(converged_files,sort_key,-1)
	sad_files=np.take_along_axis(sad_files,sort_key,-1)
	xyz_files=np.take_along_axis(xyz_files,sort_key,-1)

	#cutoff some number of timesteps if specified
	if cutoff_t:
		sad_files = sad_files[:cutoff_t]
		converged_files = converged_files[:cutoff_t]
		xyz_files = xyz_files[:cutoff_t]

	n_timesteps = len(xyz_files)

	#load the matrices
	sads_full=np.array([np.load("sad_guess/"+file) for file in sad_files])
	convergeds_full=np.array([np.load("converged_fock/"+file) for file in converged_files])

	#return the UT blocks of each matrix
	ut_block_sads=ut_block(sads_full,n_split,use_lt)
	ut_block_convergeds=ut_block(convergeds_full,n_split,use_lt)

	#convert both SADs and SCFs to supermatrices
	sad_super = to_supermat(ut_block_sads)
	scf_super = to_supermat(ut_block_convergeds)

	#make the directory to write outputs
	if not os.path.exists(dir_name):
		os.mkdir(dir_name)

	#all krr happens here
	all_predicted=loop_krr_fock(train_sizes,sad_super,scf_super,n_timesteps,dir_name,save,alpha,inner_product_type,n_split,scaling,span_c, next_t,split_alpha,subsampling)

	#solve predicted hamiltonians for total energy / mo energies
	if solve or save or full_matrix_mae:

		unfolded_all_predicted = []

		for supermat in all_predicted:
			block_size = math.sqrt(len(supermat[0]))
			ut_blocks = un_supermat(supermat,len(supermat)//n_split,n_split) 
			fullmat = unblock_ut_mats(ut_blocks,use_lt)
			unfolded_all_predicted.append(fullmat)
		unfolded_all_predicted = np.array(unfolded_all_predicted)

	if diagonal_correct:

		for s, train in enumerate(unfolded_all_predicted):
			for t, timestep in enumerate(train):
				for i in range(len(timestep)):
					for j in range(len(timestep)):
						if j>i:
							avg = (timestep[i][j] + timestep[j][i]) / 2
							unfolded_all_predicted[s][t][i][j] = avg 
							unfolded_all_predicted[s][t][j][i] = avg 


	if full_matrix_mae:
		print()
		print("Test Full Matrix MAE:", train_sizes)
		for s, train in enumerate(unfolded_all_predicted):
			all_errors = []
			for t, timestep in enumerate(train):
				for i in range(len(timestep)):
					for j in range(len(timestep)):
						all_errors.append(abs(convergeds_full[t+train_sizes[s]][i][j] - timestep[i][j]))
						
			print(round_2_sigfigs(sum(all_errors)/len(all_errors),3))

	if deep_dive:
		for s, train in enumerate(unfolded_all_predicted):
			diag_abs_errors = []
			offdiag_abs_errors = []
			pure_diag_errors = []
			all_errors = []
			delta_sym = []
			diags = []
			diag_block_offdiags = []
			offdiags = []
			ut_mae = []
			lt_mae = []
			od_delta_sym = []
			od_lt_mae = []
			od_ut_mae = []
			for t, timestep in enumerate(train):
				for i in range(len(timestep)):
					rowblock = i // block_size
					for j in range(len(timestep)):
						#if j>=i:
							colblock = j // block_size
							all_errors.append(abs(convergeds_full[t+train_sizes[s]][i][j] - timestep[i][j]))
							if rowblock == colblock:
								if i == j:
									pure_diag_errors.append(abs(convergeds_full[t+train_sizes[s]][i][j] - timestep[i][j]))
									diags.append(abs(timestep[i][j]))
								else:
									diag_abs_errors.append(abs(convergeds_full[t+train_sizes[s]][i][j] - timestep[i][j]))
									delta_sym.append(abs(timestep[i][j]-timestep[j][i]))
									diag_block_offdiags.append(abs(timestep[i][j]))
									if i > j:
										lt_mae.append(abs(convergeds_full[t+train_sizes[s]][i][j] - timestep[i][j]))
									else:
										ut_mae.append(abs(convergeds_full[t+train_sizes[s]][i][j] - timestep[i][j]))
							else:
								offdiag_abs_errors.append(abs(convergeds_full[t+train_sizes[s]][i][j] - timestep[i][j]))
								offdiags.append(abs(timestep[i][j]))
								od_delta_sym.append(abs(timestep[i][j]-timestep[j][i]))
								if rowblock < colblock:
									od_ut_mae.append(abs(convergeds_full[t+train_sizes[s]][i][j] - timestep[i][j]))
								else:
									od_lt_mae.append(abs(convergeds_full[t+train_sizes[s]][i][j] - timestep[i][j]))

			print()
			print("For train", train_sizes[s])
			print("Total MAE:", sum(all_errors)/len(all_errors))
			print()
			print("Mean Absolute Symmetry Difference, Diagonal Blocks:", sum(delta_sym)/len(delta_sym))
			print("MAE for",len(ut_mae),"diagonal block upper offdiag elements is",sum(ut_mae)/len(ut_mae))
			print("MAE for",len(lt_mae),"diagonal block lower offdiag elements is",sum(lt_mae)/len(lt_mae))
			print("Diagonal Block Off-Diagonal Element Mean Magnitude:", sum(diag_block_offdiags)/len(diag_block_offdiags))
			print("MAE for",len(diag_abs_errors),"diagonal block offdiag elements is",sum(diag_abs_errors)/len(diag_abs_errors))
			print()
			print("MAE for",len(pure_diag_errors),"diagonal  elements is",sum(pure_diag_errors)/len(pure_diag_errors))
			print("Diagonal Element Mean Magnitude:", sum(diags)/len(diags))
			print()
			if n_split > 1:
				print("MAE for",len(offdiag_abs_errors),"off-diagonal block elements is",sum(offdiag_abs_errors)/len(offdiag_abs_errors))
				print("Off-Diagonal Element Mean Magnitude:", sum(offdiags)/len(offdiags))
				print("MAE for",len(ut_mae),"off-diagonal ut block elements is",sum(ut_mae)/len(ut_mae))
				print("MAE for",len(lt_mae),"off-diagonal lt block elements is",sum(lt_mae)/len(lt_mae))
				print("Mean Absolute Symmetry Difference, Off-Diagonal Blocks:", sum(od_delta_sym)/len(od_delta_sym))
			print()

	if save:
		name=dir_name+"/predicted_converged_focks.npy"
		np.save(name,unfolded_all_predicted)

	if solve:
		if solve=="all":
			mae_tot_e,mae_mo_e=solve_fock_matrices(unfolded_all_predicted,convergeds_full,xyz_files,train_sizes)
		else:
			try:
				chosen_train=int(solve)
				chosen_ind=train_sizes.index(chosen_train)
				mae_tot_e,mae_mo_e=solve_fock_matrices(unfolded_all_predicted[chosen_ind],convergeds_full,xyz_files,chosen_train)
			except ValueError:
				print()
				print("Solving skipped: invalid value provided for 'solve' - should be a training size")
				print()

	end=time.time()
	print("Timing:",round(end-start,1),"s")

def ut_block(matrices,n_split,use_lt):
	'''
	Outputs one set of blocks per matrix, and the blocks are ouput as an array (instead of as a matrix of matrices)
	raises an exception if the matrices are not square
	Now has an option to include lower triangular blocks
	Inputs:
		a np array of matrices (a np array (timesteps) of np array (rows) x np array (columns))
	Outputs:
		a np array of ut blocked matrices (np array (timesteps) of np array (blocks) of np array (rows) x np array (columns))
	'''
	matrices=np.array(matrices)
	timesteps=[]
	roots=np.roots([1,1,-2*n_split]) # roots of x^2 + x - 2*n_split
	for matrix in matrices:

		rows,cols=matrix.shape
		if rows!=cols:
			raise Exception("Matrices are not square!")
		
		if use_lt:
			n_blocks_per_row=math.sqrt(n_split)
			block_nrow=rows/n_blocks_per_row #the block size
			if not block_nrow.is_integer():
				raise Exception("Chosen n_split does not evenly divide matrix")
			block_nrow=int(block_nrow)

			blocks=np.array([np.zeros((block_nrow,block_nrow)) for _ in range(n_split)])

			count=0
			for i in range(0, rows, block_nrow):
				for j in range(0, cols, block_nrow):
					blocks[count]=matrix[i:i+block_nrow,j:j+block_nrow]
					count=count+1

		else:
			n_blocks_per_row=[x for x in roots if x>0][0]	
			block_nrow=rows/n_blocks_per_row #the block size
			if not block_nrow.is_integer():
				raise Exception("Chosen n_split does not evenly divide matrix")
			block_nrow=int(block_nrow)

			blocks=np.array([np.zeros((block_nrow,block_nrow)) for _ in range(n_split)])

			count=0
			for i in range(0, rows, block_nrow):
				for j in range(i, cols, block_nrow):
					blocks[count]=matrix[i:i+block_nrow,j:j+block_nrow]
					count=count+1
					if use_lt and i != j:
						blocks[count]=matrix[j:j+block_nrow,i:i+block_nrow]
						count=count+1


		timesteps.append(blocks)

	return np.array(timesteps)


def to_supermat(ut_blocks):
	#turns a 4D block series (timesteps x blocks x 2D matrices) to a 2D supermatrix of (timesteps * blocks) x block element

	n_timesteps , n_blocks, n_rows, n_cols = ut_blocks.shape

	supermat = np.zeros((n_timesteps*n_blocks,n_rows*n_cols))

	for timestep, blocks in enumerate(ut_blocks):
		for block_n, block_els in enumerate(blocks):
			for row_n, row in enumerate(block_els):
				for col_n,element in enumerate(row):
					supermat[ (timestep * n_blocks) + block_n , (row_n * n_cols) + col_n] = element

	return supermat


def loop_krr_fock(train_sizes,sad_super,scf_super,n_timesteps,dir_name,save,alpha,inner_product_type,n_split,scaling,span_c, next_t,split_alpha,subsampling):
	num_workers = os.cpu_count()
	all_predicted=[]

	for train_size in train_sizes:
		if train_size<n_timesteps:

			predicted=krr_fock(train_size,sad_super,scf_super,dir_name,save,alpha,inner_product_type,n_split,scaling,span_c, next_t,split_alpha,subsampling)
			all_predicted.append(predicted)

	#all_predicted = Parallel(n_jobs = num_workers)(delayed(krr_fock)(train_size,sad_super,scf_super,dir_name,save,alpha,inner_product_type,n_split,scaling,span_c) for train_size in train_sizes if train_size < n_timesteps)

	return all_predicted


def krr_fock(train_size,sad_super,scf_super,dir_name,save,alpha,inner_product_type,n_split,scaling,span_c,next_t,split_alpha,subsampling):

	if scaling: #note that this scaling treats element a of each block the same, which may not be the best way to do this, but its worth trying
		x_scaler = StandardScaler()
		y_scaler = StandardScaler()
		sad_super = x_scaler.fit_transform(sad_super)
		scf_super = y_scaler.fit_transform(scf_super)

	cutoff_ntnb = train_size*n_split
	X_train=sad_super[:cutoff_ntnb]
	X_test=sad_super[cutoff_ntnb:]
	y_train=scf_super[:cutoff_ntnb]
	y_test=scf_super[cutoff_ntnb:]

	if next_t:
		cutoff_next_tb = next_t*n_split
		X_test = X_test[:cutoff_next_tb]
		y_test = y_test[:cutoff_next_tb]

	if subsampling:
		sample_size = int(train_size*subsampling)
		ts_indices = np.random.choice(train_size,size=sample_size,replace=False)
		indices = []
		for ts_index in ts_indices:
			for i in range(n_split):
				indices.append((ts_index*n_split) + i)
		indices = np.array(indices)
		X_train = X_train[indices]
		y_train = y_train[indices]


	#where the magic happens
	predicted = blocked_krr(X_train,X_test,y_train,alpha,inner_product_type,span_c,split_alpha,n_split)
	predict_train = blocked_krr(X_train,X_train,y_train,alpha,inner_product_type,span_c,split_alpha,n_split)
	# predicted = elementwise_blocked_krr(X_train,X_test,y_train,alpha,inner_product_type,n_split)
	# predict_train = elementwise_blocked_krr(X_train,X_train,y_train,alpha,inner_product_type,n_split)
	
	if scaling:
		y_train = y_scaler.inverse_transform(y_train)
		predicted = y_scaler.inverse_transform(predicted)
		y_test = y_scaler.inverse_transform(y_test)
		predict_train = y_scaler.inverse_transform(predict_train)

	#mae_train, mae_test = find_mae(y_train,predicted,y_test,predict_train,train_size,n_split)	

	# if save:
	# 	name=dir_name+"/predicted_converged_focks_train_"+str(train_size)+".npy"
	# 	np.save(name,predicted)

	return predicted


def blocked_krr(F_tilde,f_tilde,F,alpha,inner_product_type,span_c,split_alpha,n_split):

	#note that this implementation computes all the predicts at once as one supermatrix
	#I'm not yet sure if that is equivalent

	ntrain_nB, nb = F_tilde.shape

	ntest_nB, nb = f_tilde.shape

	#this is an assumption, I'm not sure if this is the right way to do this
	gamma_2 = (alpha**2)*np.identity(ntrain_nB)

	if inner_product_type == "classic":

		#both of these might also be wrong
		#each row of the supermatrices is a flattened block
		#so elementwise block inner products are just dot products of the rows
		#currently I compute every block product, but shouldn't only like blocks be compared?
		S_tilde = np.zeros((ntrain_nB,ntrain_nB))

		for i,row1 in enumerate(F_tilde):
			for j,row2 in enumerate(F_tilde):
				S_tilde[i,j] = np.dot(row1,row2)


		f_tilde_by_F_tilde = np.zeros((ntest_nB,ntrain_nB))

		for i,row1 in enumerate(f_tilde):
			for j,row2 in enumerate(F_tilde):
				f_tilde_by_F_tilde[i,j] = np.dot(row1,row2)

	elif inner_product_type == "damped": #the idea is to decrease the magnitude of the inner product when blocks have different block indices

		#for now, arbitrarily choosing 0.1 as the damping factor

		S_tilde = np.zeros((ntrain_nB,ntrain_nB))

		for i,row1 in enumerate(F_tilde):

			block_i = i // n_split

			for j,row2 in enumerate(F_tilde):

				block_j = j // n_split

				if block_i == block_j:

					S_tilde[i,j] = np.dot(row1,row2)

				else: 

					S_tilde[i,j] = 0.5 * np.dot(row1,row2)

		f_tilde_by_F_tilde = np.zeros((ntest_nB,ntrain_nB))

		for i,row1 in enumerate(f_tilde):

			block_i = i // n_split

			for j,row2 in enumerate(F_tilde):

				block_j = j // n_split

				if block_i == block_j:

					f_tilde_by_F_tilde[i,j] = np.dot(row1,row2)

				else:

					f_tilde_by_F_tilde[i,j] = 0.1 * np.dot(row1,row2)


	else:
		raise Exception("Other inner products not implemented")

	inv = np.linalg.inv(S_tilde + gamma_2)

	coeff = f_tilde_by_F_tilde  @ inv

	# name = str(ntrain_nB)+"_linear_coeff.npy"
	# if not os.path.exists(name):
	# 	np.save(name,coeff)

	predicted = coeff @ F

	if span_c:

		if split_alpha:
			gamma_2 = (split_alpha**2)*np.identity(ntrain_nB)
			inv = np.linalg.inv(S_tilde + gamma_2)

		coeff = f_tilde_by_F_tilde  @ inv

		f_bar =   coeff @ F_tilde

		correction = f_tilde - f_bar

		predicted = predicted + (span_c*correction)


	return predicted


def find_mae(y_train,predicted,y_test,predict_train,train_size,n_split):

	
	abs_diff_train=np.abs(predict_train-y_train)
	mae_train=np.mean(abs_diff_train)
	print("Train MAE matrix elements w/ train size",train_size,":",round_2_sigfigs(mae_train,3))

	abs_diff_test=np.abs(predicted-y_test)
	mae_test=np.mean(abs_diff_test)
	print("Test MAE matrix elements w/ train size",train_size,":",round_2_sigfigs(mae_test,3))

	return mae_train, mae_test

def round_2_sigfigs(n,sigfigs):

	return round(n,sigfigs-int(math.floor(math.log10(abs(n)))))

def un_supermat(supermat,n_timesteps,n_split):
	#the inverse of to_supermat

	_, nb = supermat.shape

	n_row_per_block = int(math.sqrt(nb))

	timesteps_blocks = np.zeros((n_timesteps,n_split,n_row_per_block,n_row_per_block))

	timestep = 0
	block = 0
	for flatblock in supermat:
		row = 0
		col = 0
		for el in flatblock:
			timesteps_blocks[timestep][block][row][col] = el

			if col+1==n_row_per_block:
				row += 1
				col = 0
			else:
				col += 1
		if block+1 == n_split:
			timestep += 1
			block = 0
		else:
			block += 1

	return timesteps_blocks

def in_prod(A,B,inner_product_type): 
	if inner_product_type=="classic":
		return np.vdot(A,B)
	else:
		raise Exception("Specified inner product not yet impleme.////,,,,,,,,,,,,,,,,,,,,,,,m..,,,,.m,,,,,,,,,,,,,,,,,mnted")

def unblock_ut_mats(blocks,use_lt):
	'''
	the inverse of the ut_block function
	Inputs:
		a np array of ut blocked matrices (np array (timesteps) of np array (blocks) of np array (rows) x np array (columns))
	Outputs:
		a np array of matrices (a np array (timesteps) of np array (rows) x np array (columns))
	'''
	blocks=np.array(blocks)
	unblocked=[]
	for timestep in blocks:
		n_blocks=len(timestep)
		if use_lt:
			n_blocks_per_row = int(math.sqrt(n_blocks))
			block_size=len(timestep[0])
			big_mat_size=int(n_blocks_per_row*block_size)
			big_mat=np.zeros((big_mat_size,big_mat_size))
			block_row=0
			block_col=0
			for block in timestep:

				big_mat[block_row : block_row + block_size , block_col : block_col + block_size] = block

				if (block_col/block_size)<n_blocks_per_row-1:
					block_col=block_col+block_size
				else:
					block_row=block_row+block_size
					block_col=0

			unblocked.append(big_mat)
		else:
			roots=np.roots([1,1,-2*n_blocks])
			n_blocks_per_row=[x for x in roots if x>0][0]
			block_size=len(timestep[0])
			big_mat_size=int(n_blocks_per_row*block_size)
			big_mat=np.zeros((big_mat_size,big_mat_size))
			block_row=0
			block_col=0
			for block in timestep:
				#print(block_row,block_col)
				big_mat[block_row : block_row + block_size , block_col : block_col + block_size] = block

				if block_row!=block_col: #off-diagonals
					big_mat[block_col : block_col + block_size , block_row : block_row + block_size] = block.T

				if (block_col/block_size)<n_blocks_per_row-1:
					block_col=block_col+block_size
				else:
					block_row=block_row+block_size
					block_col=block_row

			unblocked.append(big_mat)

	return unblocked

def elementwise_blocked_krr(F_tilde,f_tilde,F,alpha,inner_product_type,n_split):
	#replaces the matrix multiplication with good old fashioned element-wise computation
	#I assume slower, the important thing is if this gives the same answer

	ntrain_nB, nb = F_tilde.shape

	ntest_nB, nb = f_tilde.shape

	#this is an assumption, I'm not sure if this is the right way to do this
	gamma_2 = (alpha**2)*np.identity(ntrain_nB)

	if inner_product_type == "classic":

		#both of these might also be wrong
		#each row of the supermatrices is a flattened block
		#so elementwise block inner products are just dot products of the rows
		#currently I compute every block product, but shouldn't only like blocks be compared?
		S_tilde = np.zeros((ntrain_nB,ntrain_nB))

		for i,row1 in enumerate(F_tilde):
			for j,row2 in enumerate(F_tilde):
				S_tilde[i,j] = np.dot(row1,row2)

	else:
		raise Exception("Other inner products not implemented")

	M = np.linalg.inv(S_tilde + gamma_2)

	n_predicts = int(ntest_nB / n_split)

	predicted=np.zeros((ntest_nB, nb))
	for time in range(n_predicts):
		for block in range(n_split):
			for block_el in range(nb):

				time_block = (time * n_split) + block
				
				for block_el_b in range(nb):
					for timeblock_bb in range(ntrain_nB):
						for timeblock_cy in range(ntrain_nB):
							predicted[time_block][block_el] += f_tilde[time_block][block_el_b] * F_tilde[timeblock_bb][block_el_b] * M[timeblock_bb][timeblock_cy] * F[timeblock_cy][block_el]



	return predicted

def solve_fock_matrices(all_predicted,convergeds_full,xyz_files,train_sizes):
	'''
	2 modes: either a for loop over all training sizes of just for a single training size
	Determined by whether train_sizes is a list or an integer
	'''
	if isinstance(train_sizes,int):
		mae_tot_e,mae_mo_e=solve_focks(all_predicted,convergeds_full,xyz_files,train_sizes)
	else:
		mae_tot_e=[]
		mae_mo_e=[]
		for i,train_size in enumerate(train_sizes):
			chosen_ind=train_sizes.index(chosen_train)
			mae_tot_e_i,mae_mo_e_i=solve_focks(all_predicted[chosen_ind],convergeds_full,xyz_files,train_size)
			mae_tot_e.append(mae_tot_e_i)
			mae_mo_e.append(mae_mo_e_i)

	return mae_tot_e,mae_mo_e

def unfold_1d_upper_triangular(predicted_mats,mat_size):

	#need to un-fold the 1D upper triangular predicted Fock matrices
	unfolded_mats=[]
	for j,mat in enumerate(predicted_mats):

		temp_mat=np.zeros((mat_size,mat_size))
		row=0
		col=0
		for i,element in enumerate(mat):

			temp_mat[row][col]=element
			temp_mat[col][row]=element


			col=col+1
			if col==mat_size:
				row=row+1
				col=row
			
		unfolded_mats.append(temp_mat)

	return np.array(unfolded_mats)


def solve_focks(predicted_mats,convergeds_full,xyz_files,train_size):
	test_set=convergeds_full[train_size:]
	xyz_test=xyz_files[train_size:]

	matrix_size=len(convergeds_full[0])

	tot_e_maes=[]
	mo_e_maes=[]
	for i,predict in enumerate(predicted_mats):
		xyz_name="xyz_coords/"+xyz_test[i]
		mol=gto.M(atom=xyz_name, basis="def2svp")	

		mol.spin=0
		mol.charge=0
		mol.verbose=4

		reference=test_set[i]
		mf_true = dft.RKS(mol,xc="PBE")
		mf_true.max_cycle=1
		
		print()
		print("Solving Exact Fock")
		print()
		mf_true.get_fock = lambda *args: reference
		mf_true.kernel()
		total_e_true=mf_true.e_tot
		mo_es_true=mf_true.mo_energy


		print()
		print("Solving Predicted Fock")
		print()

		mf_predict= dft.RKS(mol,xc="PBE")
		mf_predict.max_cycle=1
		mf_predict.get_fock = lambda *args: predict
		mf_predict.kernel()
		total_e_predict=mf_predict.e_tot
		mo_es_predict=mf_predict.mo_energy

		tot_e_maes.append(abs(total_e_true-total_e_predict))
		mo_e_maes.append(metrics.mean_absolute_error(mo_es_predict,mo_es_true))

		print()
		print("Statistics at step",i)
		print("Total energy MAE:",round_2_sigfigs(sum(tot_e_maes)/len(tot_e_maes),3))
		print("MO energy MAE:",round_2_sigfigs(sum(mo_e_maes)/len(mo_e_maes),3))
		print()

	return sum(tot_e_maes)/len(tot_e_maes), sum(mo_e_maes)/len(mo_e_maes)




def split_arrays(array,n_split,cross,type): #unused

	ut_length=len(array[0])
	test=ut_length/n_split
	if (test*(test+1))/2 < ut_length:
		raise Exception("Too many segments chosen:",test,"is less than the side length")

	split_array=[[] for _ in range(n_split)]

	for mat in array:
		segments=np.array_split(mat,n_split)

		if cross=="more_timesteps":
			if type=="x":
				for i in range(n_split):
					for j in range(n_split):
						split_array[i].append(segments[j])
			elif type=="y":
				for i in range(n_split):
					for j in range(n_split):
						split_array[i].append(segments[i])

		else:
			for i in range(n_split):
				split_array[i].append(segments[i])

	print(np.array(split_array).shape)	

	return split_array

def split_krr_fock(pipe,X_train,y_train,X_test,cross):

	for i,segment in enumerate(X_train):
		pipe.fit(segment,y_train[i])
		predicted_i=pipe.predict(X_test[i])
		if i>0:
			predicted=[np.concatenate([prev, new], axis=0) for prev, new in zip(predicted, predicted_i)]
		else:
			predicted=predicted_i		

	predicted=np.array(predicted)

	if cross=="more_timesteps":
		n_segments=len(X_test)
		n_timesteps=int(len(predicted)/n_segments)
		len_mat=len(predicted[0])
		#print(n_segments,len(predicted),len_mat)
		reshaped=predicted.reshape(n_timesteps,n_segments,len_mat)
		averaged=reshaped.mean(axis=1)
		predicted=np.copy(averaged)

	return predicted





if __name__ == "__main__":
	main()