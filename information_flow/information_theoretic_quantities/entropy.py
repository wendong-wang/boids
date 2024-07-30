#author: Pang Jiahuan
#start: 2023/6/13

import numpy as np
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt


class Distribution():
    def __init__(self, pmf:np.ndarray, rv_name:list ) -> None:
        '''
        joint distribution

        Parameters
        ---
        pmf: joint distribution (use matrix to represent it )
        rv_name: random varibles' name 
        '''
        self.pmf = pmf # joint distribution
        self.rv_name = rv_name # random variables' names
        self.rv_number = np.shape(rv_name)[0] # the number of variables 
        self.rv_list = list(np.arange(self.rv_number))# !!!temp: use numbers to denotes the random variables


    def marginal_distribution(self,rv:list)->np.ndarray:
        sum_list = self.rv_list[:]
        for i in rv:
            rvIndex = self.rv_name.index(i)
            sum_list.remove(rvIndex)# 

        mdis = np.sum(self.pmf,axis=tuple(sum_list))
        return mdis
    
    def conditioned_distribution(self, crv:list, rv:list)->np.ndarray:
        '''
        # Parameters
        crv: the conditioning variables 
        rv: the variables be conditioned

        # Example
        condition_distribution(crv = ['X','Z'], rv = ['Y']) -> p(y|x,z)
        '''
        join_rv_Name = self.rv_name[:]
        for i in join_rv_Name:
            if i not in (crv + rv):
                join_rv_Name.remove(i)

        # obtain the related variables's joint distribution
        joint_distribution = Distribution(pmf=self.marginal_distribution(rv = crv+rv), rv_name= join_rv_Name)
        crv_distribution = joint_distribution.marginal_distribution(rv = crv)
        
        #reshape the crv_distribution to match joint_distribution
        reshape_tuple = np.ones(joint_distribution.rv_number,dtype=np.int8)
        for i in crv:
            rvIndex = joint_distribution.rv_name.index(i)
            reshape_tuple[rvIndex] = np.shape(joint_distribution.pmf)[rvIndex] # for (crv = ['X','Z'], rv = ['Y']) cases, reshape_tuple=(n,1,n)



        crv_distribution = crv_distribution.reshape(tuple(reshape_tuple))

        mask = np.ones(np.shape(joint_distribution.pmf))*crv_distribution
        
        # print(joint_distribution.pmf,"\n",crv_distribution)
        conditioned_distribution = np.divide(joint_distribution.pmf, crv_distribution, where= mask>0 ) 
                                             
        return conditioned_distribution
        

def entropy(dis:Distribution, rv:list):
    '''
    calculate entropy
    
    Parameter
    ---
    dis: the joint distribution
    rv: the variable's name
    '''
    #obtain the distribution of the virable
    vdis = dis.marginal_distribution(rv)

    # calculation the entropy
    entropy_result = 0
    entropy_result = np.sum([-1*i*np.log2(i) for i in vdis.flat if i!=0])

    return entropy_result


def joint_entropy(dis:Distribution, rv:list):
    '''
    calculate the joint entropy
    
    Parameter
    ---
    dis: the joint distribution
    rv: the variables' name needs to joining
    '''

    #calculate the joint entropy
    joint_entropy_result = 0
    joint_entropy_result = entropy(dis=dis,rv=rv)
    return joint_entropy_result





def conditioned_enrtopy(dis:Distribution, rv:list, crv:list):
    '''
    calculate the conditioned entropy

    use the chain rule for entropy to achieve this calculation:
    H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)

    Parameters
    ---
    dis: the joint distribution
    rv: random variable
    crv: the conditioned variable

    '''
    conditioned_entropy_result = 0

    conditioned_entropy_result = joint_entropy(dis,rv+crv) - joint_entropy(dis,crv)
    return conditioned_entropy_result

def mutual_infomration(dis:Distribution, rv1:list, rv2:list):
    '''
    calculate the mutual information
    I(X;Y) = H(X) - H(X|Y) = H(X) + H(Y) - H(X,Y)
    focus on two variable

    Parameters
    ---
    dis: joined distribution
    rv1 & rv2: the two sides of mutual information I(rv1;rv2) 

    '''

    mutual_infomration_result = 0



    mutual_infomration_result = joint_entropy(dis, rv1)+joint_entropy(dis,rv2)-joint_entropy(dis,rv1+rv2)

    return mutual_infomration_result


def conditioned_mutual_information(dis:Distribution, rv1:list, rv2:list, crv:list):
    '''
    calculate the conditioned mutual information 
    use chain rule for mutual information 
    I(X;Y,Z) = I(X;Y)+I(X;Z|Y) = I(X;Z) + I(X;Y|Z)

    '''
    conditioned_mutual_information_result=0
    
    conditioned_mutual_information_result = mutual_infomration(dis,rv1,rv2+crv) - mutual_infomration(dis,rv1,crv)
    
    return conditioned_mutual_information_result



def obtain_a_transition_matrix(matrix: np.ndarray):
    '''
    Obtain a conditioned distriution
    the conditioned variable is in the first dimension
    '''

    crv_dim = int(np.sqrt(np.shape(matrix)[0]))
    # transport_matrix = np.abs(matrix) 
    transport_matrix = matrix.reshape(crv_dim,crv_dim)

    # normalize for every row
    margin_sum =  np.sum(transport_matrix,axis=1) #in case of all zero
    transport_matrix = (transport_matrix + (margin_sum ==0 ).reshape(crv_dim,1))

    margin_sum =  np.sum(transport_matrix,axis=1)
    # margin_sum = np.array([x if x>0 else x+0.0001 for x in margin_sum ]).reshape(crv_dim,1)
    margin_sum =  margin_sum.reshape(crv_dim,1)

    ##get the transport matrix
    transport_matrix = transport_matrix/margin_sum
    return transport_matrix






def IMI_candidate(optimize_matrix,*args):

    
    # global basin_hopping_result_record
    # normalization    
    dis = args[0]

    optimize_matrix = obtain_a_transition_matrix(optimize_matrix)


    new_joint_dis = dis.pmf@optimize_matrix
    new_dis = Distribution(pmf=new_joint_dis,rv_name=["X","Y","Z_bar"])
 
    result = conditioned_mutual_information(new_dis,["X"],["Y"],["Z_bar"])
    # basin_hopping_result_record.append(result)
    return  result



def intrinsic_mutual_information(dis:Distribution, rv1:list, rv2:list, crv:list):
    '''
    calculate the intrinsic mutual information 
    only for three variables cases and the conditioned one should in the last dimension

    I(X_{-1};Y_{0}|\bar{Y_{-1}})

    I(X;Y|Z_bar)

    I(rv1;rv2|\bar{crv})

    Parameter
    ---
    dis: the joint distribution
    crv: the conditioned random variable

    '''

    
    crvIndex = 2
    crv = dis.rv_name[2]

    
    # initialize the transport matrix P(Z,\bar{Z})
    pmf_size = np.shape(dis.pmf)
    opInitMatrix = np.ones((pmf_size[crvIndex]**2))/np.sum(np.ones((pmf_size[crvIndex]**2))) # 1
    
    #calculate the margin distribution of the conditioned random variable 
    crv_dis = dis.marginal_distribution(crv)

    bnds = np.dstack((np.zeros((pmf_size[crvIndex]**2)),np.ones((pmf_size[crvIndex]**2))))
    bnds = tuple(tuple(i) for i in bnds[0])
    # print(bnds)
    # minimizer_kwargs  = {"args": (dis,crv_dis), "bounds": bnds}
    minimizer_kwargs  = {"args": (dis,crv_dis),"bounds": bnds}
    ret = basinhopping(IMI_candidate, x0=opInitMatrix, minimizer_kwargs=minimizer_kwargs, stepsize=0.5, niter=5)

    return ret

#===========================================================================================================

def specific_information(dis: Distribution, rv_receievers: list, sources: list):
    '''
    calculate the specific information 
    I(A;S=s) = \sum_a p(a|s)(log(1/p(s)) - log(1/p(s|a))) = \sum_a p(a|s)(log(1/ p(a)) - log(1/p(a|s))) 

    Parameter
    ---
    dis: the joint distribution
    rv_receievers: the varible that information pass to (only one)
    rv_receievers_value: S=s
    sources : the  sources 
    '''
    # see the conditioned distribution part
    join_rv_Name = dis.rv_name[:]
    for i in join_rv_Name:
        if i not in (sources + rv_receievers):
            join_rv_Name.remove(i)

    joint_distribution = Distribution(pmf=dis.marginal_distribution(rv = sources + rv_receievers), rv_name= join_rv_Name)
    
    # obtain the reshape tuple 
    reshape_tuple = np.ones(joint_distribution.rv_number,dtype=np.int8)
    for i in sources: 
        rvIndex = joint_distribution.rv_name.index(i)
        reshape_tuple[rvIndex] = np.shape(joint_distribution.pmf)[rvIndex]


    
    conditioned_dis_on_s = joint_distribution.conditioned_distribution(rv = sources, crv = rv_receievers)
    # print(conditioned_dis_on_s)
    
    margin_dis_a = joint_distribution.marginal_distribution(rv = sources)

    
    margin_dis_a = margin_dis_a.reshape(tuple(reshape_tuple))

    # calculate
    log_conditioned_s = np.log2(conditioned_dis_on_s, where=conditioned_dis_on_s!=0) ##
    log_on_a = np.log2(margin_dis_a, where=margin_dis_a!=0).reshape(tuple(reshape_tuple))
    substract_result = (log_conditioned_s-log_on_a)
    sp_in =  conditioned_dis_on_s*substract_result

    sum_list = joint_distribution.rv_list[:]
    # print("hello",sum_list)
    for i in rv_receievers: #sum on the sources so remove the receiver
        rvIndex = joint_distribution.rv_name.index(i)
        sum_list.remove(rvIndex)

    _specific_information = np.sum(sp_in, axis=tuple(sum_list))


    return _specific_information





def redundancy_information(dis: Distribution, rv_receievers:list, sources_list:list):
    '''
    calculate the redundancy information of a sources list 


    Parameter
    ---
    dis: the joint distribution
    rv_receievers: the varible that information pass to
    sources list: sources list {1},{2} or {{1},{2}}. list of list
    '''

    # candidate list
    candidate_list = []
    rv_receievers_dis = dis.marginal_distribution(rv=rv_receievers)  # only for one dimension
    redundancy_information = 0
    
    
    for sources in sources_list:
        candidate_list.append(specific_information(dis=dis, rv_receievers=rv_receievers, sources=sources))
    
    for i in range(np.shape(rv_receievers_dis)[0]):
        redundancy_information += rv_receievers_dis[i] * np.min(np.array(candidate_list)[:,i])


    
    return redundancy_information




def partial_information(dis: Distribution, rv_receievers:list, rv_senders:list)->tuple:
    '''
    calculate the PI-function (unique, redundancy, synergistic)
    3 virables


    Parameter
    ---
    dis: the joint distribution
    rv_receievers: the varible that information pass to
    rv_senders: the varibles that give information

    return
    ---
    unique_information_sender1, unique_information_sender2, _redundancy_information, synergistic_information
    '''
    #redundancy of {1}{2} which is the redundancy information
    _redundancy_information = redundancy_information(dis = dis, rv_receievers = rv_receievers, sources_list = [[rv_senders[0]],[rv_senders[1]]] ) ###!!!!!!!!!!!!!1

    #calculate the unique information of {1} and {2}
        #first mutual information belongs to {1} and {2}
    mutual_infomration_sender1 = mutual_infomration(dis=dis,rv1=rv_receievers, rv2=[rv_senders[0]])
    mutual_infomration_sender2 = mutual_infomration(dis=dis,rv1=rv_receievers, rv2=[rv_senders[1]])
        #get the redundancy belongs to {1} and {2} which are the unique information 
    unique_information_sender1 = mutual_infomration_sender1 - _redundancy_information ##
    unique_information_sender2 = mutual_infomration_sender2 - _redundancy_information ##


    # calulate the synergistic information of {1 2}
        #first mutual information belongs to {1 2}
    mutual_infomration_sender12 = mutual_infomration(dis=dis,rv1=rv_receievers, rv2=rv_senders)
        #get the redundancy belongs to {1 2} which is the synergistic information
    synergistic_information = mutual_infomration_sender12 - unique_information_sender1 - unique_information_sender2 - _redundancy_information

    return unique_information_sender1, unique_information_sender2, _redundancy_information, synergistic_information


if __name__=="__main__":

    basin_hopping_result_record = []
    # print(" intrinsic situation")

    # inMatrix = np.zeros((2,2,2))
    # inMatrix[0,0,0]=1/4
    # inMatrix[0,0,1]=1/4
    # inMatrix[1,1,0]=1/4
    # inMatrix[1,1,1]=1/4
    # intrisic = Distribution(inMatrix, ['X','Y','Z'])

    # print("time-delayed mutual information:",mutual_infomration(intrisic,["X"],["Y"]))
    # print("transfer entropy:",conditioned_mutual_information(intrisic,["X"],["Y"],["Z"]))
    # print("intrinsic:",intrinsic_mutual_information(intrisic,["X"],["Y"],["Z"]).fun)
    # print("PI (unique_information X, unique_information Z, redundancy_information, synergistic_information): ", partial_information(dis=intrisic,rv_receievers=["Y"],rv_senders=["X","Z"]))
    
    
    
    
    # print("\n shared situation")

    # sharedMatrix = np.zeros((2,2,2))
    # sharedMatrix[0,1,0]=1/2
    # sharedMatrix[1,0,1]=1/2

    # shared = Distribution(sharedMatrix, ['X','Y','Z'])

    # print("time-delayed mutual information:",mutual_infomration(shared,["X"],["Y"]))
    # print("transfer entropy:",conditioned_mutual_information(shared,["X"],["Y"],["Z"]))
    # print("intrinsic:",intrinsic_mutual_information(shared,["X"],["Y"],["Z"]).fun)
    # print("PI (unique_information X, unique_information Z, redundancy_information, synergistic_information): ", partial_information(dis=shared,rv_receievers=["Y"],rv_senders=["X","Z"]))



    # print("\n synergistic situation")
    # synergisticMatrix = np.zeros((2,2,2))
    # synergisticMatrix[0,0,0]=1/4
    # synergisticMatrix[0,1,1]=1/4
    # synergisticMatrix[1,0,1]=1/4
    # synergisticMatrix[1,1,0]=1/4
    # synergistic = Distribution(synergisticMatrix, ['X','Y','Z'])

    # print("time-delayed mutual information:",mutual_infomration(synergistic,["X"],["Y"]))
    # print("transfer entropy:",conditioned_mutual_information(synergistic,["X"],["Y"],["Z"]))
    # print("intrinsic:",intrinsic_mutual_information(synergistic,["X"],["Y"],["Z"]).fun)
    # print("PI (unique_information X, unique_information Z, redundancy_information, synergistic_information): ", partial_information(dis=synergistic,rv_receievers=["Y"],rv_senders=["X","Z"]))
    
    #=============================================================
    # intrinsic part

    # forcus on the influence X gives to Y exclude the influence of Z
    #=============================================================

    # '''
    # example 1
    # from: Nonnegative Decomposition of Multivariate Information
    # '''
    # print("example 1:")
    # pmfM = np.zeros((3,3,3))
    # pmfM[0,0,0]=1/3
    # pmfM[0,1,1]=1/3
    # pmfM[1,2,0]=1/3
    # pmf = Distribution(pmfM, ['X','Y','Z'])
    
    # # print("time-delayed mutual information:",mutual_infomration(pmf,["X"],["Y"]))
    # # print("transfer entropy:",conditioned_mutual_information(pmf,["X"],["Y"],["Z"]))
    # print("intrinsic:",intrinsic_mutual_information(pmf,["X"],["Y"],["Z"]).fun)
    # print("PI (unique_information X, unique_information Z, redundancy_information, synergistic_information): ",partial_information(dis=pmf,rv_receievers=["Y"],rv_senders=['X','Z']))
    # print("\n")
    
    
    # '''
    # example 2
    # from: Nonnegative Decomposition of Multivariate Information
    # '''
    # print("example 2:")
    # pmfM = np.zeros((3,3,3))
    # pmfM[0,0,0]=1/3
    # pmfM[1,1,0]=1/3
    # pmfM[0,2,1]=1/3
    # pmf = Distribution(pmfM, ['X','Y','Z'])
    
    # print("intrinsic:",intrinsic_mutual_information(pmf,["X"],["Y"],["Z"]).fun)
    # print("PI (unique_information X, unique_information Z, redundancy_information, synergistic_information): ",partial_information(dis=pmf,rv_receievers=["Y"],rv_senders=['X','Z']))

    # print("\n")
    # '''
    # example 3 
    # from: Unique Information and Secret Key Agreement
    # Pintwise Unique
    # '''

    # print("example 3 (Pintwise Unique):")
    # var_alphabet_size = 3
    # pmfM = np.zeros((var_alphabet_size,var_alphabet_size,var_alphabet_size))
    # pmfM[0,1,1]=1/4
    # pmfM[1,1,0]=1/4
    # pmfM[0,2,2]=1/4
    # pmfM[2,2,0]=1/4
    # pmf = Distribution(pmfM, ['X','Y','Z'])
    # print("intrinsic:",intrinsic_mutual_information(pmf,["X"],["Y"],["Z"]).fun)
    # print("PI (unique_information X, unique_information Z, redundancy_information, synergistic_information): ",partial_information(dis=pmf,rv_receievers=["Y"],rv_senders=['X','Z']))
    
    
    # print("\n")
    # '''
    # example 4 
    # from: Unique Information and Secret Key Agreement
    # Problem
    # '''

    # print("example 4 (Problem):")
    # var_alphabet_size = 3
    # pmfM = np.zeros((var_alphabet_size,var_alphabet_size,var_alphabet_size))
    # pmfM[0,0,0]=1/4
    # pmfM[0,1,1]=1/4
    # pmfM[0,0,2]=1/4
    # pmfM[1,1,0]=1/4
    # pmf = Distribution(pmfM, ['X','Y','Z'])
    # print("intrinsic:",intrinsic_mutual_information(pmf,["X"],["Y"],["Z"]).fun)
    # print("PI (unique_information X, unique_information Z, redundancy_information, synergistic_information): ",partial_information(dis=pmf,rv_receievers=["Y"],rv_senders=['X','Z']))
    # print("\n")
    # '''
    # example 5
    # from: Unique Information and Secret Key Agreement
    # And
    # '''
    # print("example 5 (And):")
    # var_alphabet_size = 2
    # pmfM = np.zeros((var_alphabet_size,var_alphabet_size,var_alphabet_size))
    # pmfM[0,0,0]=1/4
    # pmfM[0,0,1]=1/4
    # pmfM[1,0,1]=1/4
    # pmfM[1,1,1]=1/4
    # pmf = Distribution(pmfM, ['X','Y','Z'])
    # print("intrinsic:",intrinsic_mutual_information(pmf,["X"],["Y"],["Z"]).fun)
    # print("PI (unique_information X, unique_information Z, redundancy_information, synergistic_information): ",partial_information(dis=pmf,rv_receievers=["Y"],rv_senders=['X','Z']))
    # print("\n")
    # '''
    # example 6
    # from: Unique Information and Secret Key Agreement
    # Diff
    # '''
    # print("example 6 (Diff):")
    # var_alphabet_size = 2
    # pmfM = np.zeros((var_alphabet_size,var_alphabet_size,var_alphabet_size))
    # pmfM[0,0,0]=1/4
    # pmfM[0,1,0]=1/4
    # pmfM[0,0,1]=1/4
    # pmfM[1,1,0]=1/4
    # pmf = Distribution(pmfM, ['X','Y','Z'])
    # print("intrinsic:",intrinsic_mutual_information(pmf,["X"],["Y"],["Z"]).fun)
    # print("PI (unique_information X, unique_information Z, redundancy_information, synergistic_information): ",partial_information(dis=pmf,rv_receievers=["Y"],rv_senders=['X','Z']))
    # print("\n")
    # '''
    # example 7
    # from: Unique Information and Secret Key Agreement
    # Not two
    # '''
    # print("example 7 (Not two):")
    # var_alphabet_size = 2
    # pmfM = np.zeros((var_alphabet_size,var_alphabet_size,var_alphabet_size))
    # pmfM[0,0,0]=1/5
    # pmfM[0,1,0]=1/5
    # pmfM[0,0,1]=1/5
    # pmfM[1,0,0]=1/5
    # pmfM[1,1,1]=1/5
    # pmf = Distribution(pmfM, ['X','Y','Z'])
    # print("intrinsic:",intrinsic_mutual_information(pmf,["X"],["Y"],["Z"]).fun)
    # print("PI (unique_information X, unique_information Z, redundancy_information, synergistic_information): ",partial_information(dis=pmf,rv_receievers=["Y"],rv_senders=['X','Z']))
    # # '''
    # # example 8
    # # from: Unique Information and Secret Key Agreement
    # # Two bit copy
    # # '''
    # print("\n")  
    # print("example 8 (Two bit copy):")
    # var_alphabet_size = 4
    # pmfM = np.zeros((var_alphabet_size,var_alphabet_size,var_alphabet_size))
    # pmfM[0,0,0]=1/4
    # pmfM[0,1,1]=1/4
    # pmfM[1,2,0]=1/4
    # pmfM[1,3,1]=1/4
    # pmf = Distribution(pmfM, ['X','Y','Z'])
    # print("intrinsic:",intrinsic_mutual_information(pmf,["X"],["Y"],["Z"]).fun)
    # print("PI (unique_information X, unique_information Z, redundancy_information, synergistic_information): ",partial_information(dis=pmf,rv_receievers=["Y"],rv_senders=['X','Z']))
    #================================
    # plt.figure()
    # x_size = np.shape(basin_hopping_result_record)[0]
    # x = np.arange(x_size)
    # plt.plot(x, basin_hopping_result_record)
    
    # plt.show()

    # var_alphabet_size = 3
    # pmfM = np.zeros((var_alphabet_size,var_alphabet_size,var_alphabet_size))
    # pmfM[0,0,0]=1/9
    # pmfM[1,0,2]=1/9
    # pmfM[2,0,1]=1/9
    # pmfM[1,1,1]=1/9
    # pmfM[0,1,2]=1/9
    # pmfM[2,1,0]=1/9
    # pmfM[2,2,2]=1/9
    # pmfM[0,2,1]=1/9
    # pmfM[1,2,0]=1/9
    # pmf = Distribution(pmfM, ['X','Y','Z'])
    # print("PI (unique_information X, unique_information Z, redundancy_information, synergistic_information): ",partial_information(dis=pmf,rv_receievers=["Y"],rv_senders=['X','Z']))
    # var_alphabet_size = 4
    # pmfM = np.zeros((var_alphabet_size,var_alphabet_size,var_alphabet_size))
    # pmfM[0,0,0]=1/6
    # pmfM[1,0,1]=1/6
    # pmfM[3,0,3]=1/6
    # pmfM[1,2,1]=1/6
    # pmfM[3,2,3]=1/6
    # pmfM[2,2,2]=1/6

    # pmf = Distribution(pmfM, ['X','Y','Z'])
    # print("PI (unique_information X, unique_information Z, redundancy_information, synergistic_information): ",partial_information(dis=pmf,rv_receievers=["Y"],rv_senders=['X','Z']))
# =======================================
# exam the problem distribution
# =======================================
    '''
    example 4 
    from: Unique Information and Secret Key Agreement
    Problem
    '''

    print("example 4 (Problem):")
    var_alphabet_size = 3
    pmfM = np.zeros((var_alphabet_size,var_alphabet_size,var_alphabet_size))
    pmfM[0,0,0]=1/4
    pmfM[0,1,1]=1/4
    pmfM[0,0,2]=1/4
    pmfM[1,1,0]=1/4
    pmf = Distribution(pmfM, ['X','Y','Z'])
    print("intrinsic:",intrinsic_mutual_information(pmf,["X"],["Y"],["Z"]).fun)
    print("PI (unique_information X, unique_information Z, redundancy_information, synergistic_information): ",partial_information(dis=pmf,rv_receievers=["Y"],rv_senders=['X','Z']))
    print("\n")
    
    
    pmfM = np.zeros((var_alphabet_size,var_alphabet_size,var_alphabet_size))
    pmfM[0,0,0]=1/4
    pmfM[1,1,0]=1/4
    pmfM[2,0,0]=1/4
    pmfM[0,1,1]=1/4
    pmf = Distribution(pmfM, ['X','Y','Z'])
    print("intrinsic:",intrinsic_mutual_information(pmf,["X"],["Y"],["Z"]).fun)
    print("PI (unique_information X, unique_information Z, redundancy_information, synergistic_information): ",partial_information(dis=pmf,rv_receievers=["Y"],rv_senders=['X','Z']))