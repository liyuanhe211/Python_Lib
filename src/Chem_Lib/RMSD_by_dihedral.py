# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import sys
import pathlib
from Python_Lib.My_Lib import *
from .Lib import *
import pickle
import community

def generate_dihedral_diff_matrix(list_of_coordinates:list,bond_orders,threshold, ignore_hydrogens=True,multithread=False):
    '''

    :param list_of_coordinates:
    :param bond_orders:
    :param threshold:
    :param ignore_hydrogens:
    :param multithread:
    :return:  a matrix, each element is a list of different dihedrals that has a delta>threshold
    '''

    if multithread:
        from multiprocessing import Pool
        from multiprocessing import cpu_count

        pool = Pool(processes=int(cpu_count()/2))

    print("Generating dihedrals...")
    last_percentage=0
    print("0%")

    #分成n份计算，每份所含数量为线程数的两倍，即每个CPU算完两个后报告一次进度
    dihedrals=[]
    if multithread:
        for i in range(int(len(list_of_coordinates)/cpu_count()+1)):
            sub_list_of_coordinate = list_of_coordinates[cpu_count()*i:cpu_count()*(i+1)]
            dihedrals+= pool.map(get_dihedrals,((coordinate,bond_orders,ignore_hydrogens,False,True) for coordinate in sub_list_of_coordinate))
            current_percentage = int(i/(int(len(list_of_coordinates)/cpu_count()+1))*100)
            if current_percentage!=last_percentage:
                print(current_percentage,'%',sep="")
                last_percentage=current_percentage
    else:
        for i in range(len(list_of_coordinates)):
            dihedrals.append(get_dihedrals(list_of_coordinates[i],bond_orders,ignore_hydrogens,False,True))
            current_percentage = int(i/len(list_of_coordinates)*100)
            if current_percentage!=last_percentage:
                print(current_percentage,'%',sep="")
                last_percentage=current_percentage


    print("Generating dihedrals difference matrix...")

    last_percentage=0
    print("0%")

    diff_count_matrix =[[0 for x in range(len(dihedrals))] for x in range(len(dihedrals))]
    for row in range(len(dihedrals)):
        current_percentage = int(((len(dihedrals)*2-row)*row)/len(dihedrals)**2*100)
        if current_percentage!=last_percentage:
            print(current_percentage,'%',sep="")
            last_percentage=current_percentage
        if multithread:
            diff_list = pool.map(get_one_diff_list,((dihedrals[row],dihedrals[column],threshold) for column in range(row,len(dihedrals))))
        else:
            diff_list = [get_one_diff_list(dihedrals[row],dihedrals[column],threshold) for column in range(row,len(dihedrals))]
        for column in range(row,len(list_of_coordinates)):
            diff_count_matrix[row][column] = len(diff_list[column-row])
            diff_count_matrix[column][row] = len(diff_list[column-row])

    print("Dihedrals difference matrix generated.")
    # for i in ret_matrix:
    #     print(i)
    return diff_count_matrix

def generate_dihedral_diff_graph(diff_list_matrix):
    '''
    生成一个标准图结构，图的棱权重为棱连接两节点之间的相似度，相似度越高，棱值越小（距离越短）
    楞值=相同dihedral数目的平方
    :return:
    '''

    import networkx as nx
    G=nx.Graph()

    print("Generaing graph...")

    #仅保留40%最小的距离，其他的不设置edge，认为无穷远，节省计算量和内存
    total = len(diff_list_matrix)**2
    threshold = -1
    maximum_diff = max([max(row) for row in diff_list_matrix])
    current = 0
    if total<1000000:
        threshold = maximum_diff
        print("Will use less than 1M edges, no edge selection approximation required.")
        print("Maximum edge distance:",maximum_diff)
    else:
        for i in range(maximum_diff+1):
            print("Checking edge infinite threshold",i,"...")
            for row in diff_list_matrix:
                current+=row.count(i)
            print("Threshold ",i,"accounts for",str(int(current/total*100))+'%','results.')
            if current/total>0.4 and current>1000000:
                print("Setting threshold to",i,"accounting for",str(int(current/total*100))+'%','results.')
                threshold = i
                break
            if threshold!=-1:
                break

    last_percentage=0
    print("0%")
    added_nodes = []
    # weighted_edges = [] # something like [(node1, node2, weight)(1,2,0.125),(1,3,0.75),(2,4,1.2),(3,4,0.375)]

    total_edges = 0

    for row in range(len(diff_list_matrix)):
        current_percentage = int(((len(diff_list_matrix)*2-row)*row)/len(diff_list_matrix)**2*100)
        if current_percentage!=last_percentage:
            print(current_percentage,'%',sep="")
            last_percentage=current_percentage

        for column in range(row):
            diff_count = diff_list_matrix[row][column]
            if diff_count<=threshold:
                G.add_edge(row,column,weight=diff_count**2)
                added_nodes.append(row)
                added_nodes.append(column)
                total_edges+=1
                if total_edges%100000==0:
                    print("Total edges count:",total_edges)

    #有的节点沦落到跟谁都不接着的地位
    for node in range(len(diff_list_matrix)):
        if node not in added_nodes:
            G.add_node(node)

    print("Total edges count:",total_edges)

    # G.add_weighted_edges_from(weighted_edges)
    return G


def get_one_diff_list(dihedrals1,dihedrals2=None,threshold=None):

    if isinstance(dihedrals1, tuple) and dihedrals2==None and threshold==None:
        # 从pool.map传过来的
        dihedrals1,dihedrals2,threshold=dihedrals1

    # return the number of different between two dihedrals, if the different value > threshold
    assert len(dihedrals1)==len(dihedrals2), "Length of dihedrals mismatch."
    ret=[]
    for count,dihedral1 in enumerate(dihedrals1):
        dihedral2=dihedrals2[count]
        if dihedral1-dihedral2>threshold:
            ret.append(count)

    return ret


if __name__ =="__main__":

    assert len(sys.argv) in [4,5], "Number of Parameters Error"
    input_filename = sys.argv[1]
    which_mission = sys.argv[2] #generate diff_graph or community_dict
    output_filename = sys.argv[3]
    if which_mission=='community_dict':
        assert len(sys.argv)==5, "Number of Parameters Error"
        given_resolution_power = float(sys.argv[4]) # the actual resolution will be 10**given_resolution_power
    elif which_mission=='diff_graph':
        assert len(sys.argv)==4, "Number of Parameters Error"
    else:
        raise MyException("Which_Mission_Flag ERROR...")

    if which_mission=='diff_graph':
        with open(input_filename,'rb') as pickle_file:
            diff_matrix = pickle.load(pickle_file)
        dihedral_diff_graph = generate_dihedral_diff_graph(diff_matrix)
        with open(output_filename,'wb') as pickle_file:
            pickle.dump(dihedral_diff_graph,pickle_file)

    elif which_mission=='community_dict':
        with open(input_filename,'rb') as pickle_file:
            dihedral_diff_graph = pickle.load(pickle_file)
            output_dict = community.best_partition(dihedral_diff_graph,resolution=10**given_resolution_power)
        with open(output_filename,'wb') as pickle_file:
            pickle.dump(output_dict,pickle_file)



