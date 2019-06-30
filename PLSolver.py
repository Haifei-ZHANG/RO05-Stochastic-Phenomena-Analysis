# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 2019
@author: haifei-zhang
This is a simple solver for linear programming problem

"""
import numpy as np
import pandas as pd
import re

class PL(object):
# =============================================================================
# PL类的属性：
# 决策函数 self.objective_function 字符串
# 决策变量 self.decision_variables 字典
# 代价系数 self.cost 字典
# 优化目标 self.optimal_object 字符串
# 约束条件的初始系数表 self.initial.coefficient_matrix 数据帧
# 约束条件的系数矩阵  self.coefficient_matrix 数据帧
# 自由项 放在系数矩阵中最后一列，列名为“b”
# 检验数 放在系数矩阵中最后一行，索引为“rc”
# 可行基 self.feasible_base 列表
#可行基也可以通过系数矩阵的索引得到，出去索引的最后一个（rc）即可
# 对偶解
#测试用的规划问题    
#max 3x1+x2+3x3
#c1:2x1+x2+x3<=2
#c2:x1+2x2+3x3<=5
#c3:2x1+2x2+x3<=6
# =============================================================================

    def __init__(self):
        self.objective_function = \
        input("Please enter the objective function(ex:x1+2x2+3x3):")
        #从决策函数构建cost向量
        self.objective_function = ' ' + self.objective_function
        self.objective_function = re.sub(r'[+]x','+1x',self.objective_function)
        self.objective_function = re.sub(r'[-]x','-1x',self.objective_function)
        self.objective_function = re.sub(r'[\s]x',' 1x',self.objective_function)
        cost = re.findall(r'[+-][1-9]\d*\.?\d*|0\.?\d*[1-9]\d*$|[\s][1-9]\d*\.?\d*|0\.?\d*[1-9]\d*$',self.objective_function)
        cost = np.array(list(map(float, cost)))
        
        
        #从决策函数构建决策变量并构建cost字典
        d_variables = re.findall(r'[x]\d+',self.objective_function)
        self.decision_variables = {}
        self.cost = {}
        for variable in d_variables:
            self.decision_variables[variable] = 0
            self.cost[variable] = cost[d_variables.index(variable)]

        #确定优化目标
        self.optimal_object = \
        input("Please enter Max or Min the objective function:\t")
        if(self.optimal_object=="min" or self.optimal_object=="Min"):
            self.cost = self.cost*(-1)
        
        #确定约束条件的的个数
        self.constrains = {}
        number_constains = \
        int(input("Number of constraints："))

        #建立系数矩阵   
        coefficient_matrix_columns = list(self.decision_variables.keys())
        coefficient_matrix_columns.append("b")
        self.coefficient_matrix = pd.DataFrame(np.zeros([number_constains+1,\
        len(coefficient_matrix_columns)]), columns=coefficient_matrix_columns)
        
        for i in range(number_constains):
            self.constrains['c'+str(i+1)] = input('c'+str(i+1)+":")
            self.constrains['c'+str(i+1)] = ' ' + self.constrains['c'+str(i+1)]
            self.constrains['c'+str(i+1)] = re.sub(r'[+]x','+1x',self.constrains['c'+str(i+1)])
            self.constrains['c'+str(i+1)] = re.sub(r'[-]x','-1x',self.constrains['c'+str(i+1)])
            self.constrains['c'+str(i+1)] = re.sub(r'[\s]x',' 1x',self.constrains['c'+str(i+1)])
            coefficient_i = re.findall(r'[+-][1-9]\d*\.?\d*|0\.?\d*[1-9]\d*$|[\s][1-9]\d*\.?\d*|0\.?\d*[1-9]\d*$',self.constrains['c'+str(i+1)])
            coefficient_i = np.array(list(map(float, coefficient_i)))
            c_variables = re.findall(r'[x]\d+',self.constrains['c'+str(i+1)])
            operator = re.findall(r'[><]?=[1-9]\d*\.?\d*|0\.?\d*[1-9]\d*$',self.constrains['c'+str(i+1)])
            b_i = re.findall(r'[1-9]\d*\.?\d*|0\.?\d*[1-9]\d*$',operator[0])
            b_i = float(b_i[0])
            operator = re.sub(r'[1-9]\d*\.?\d*|0\.?\d*[1-9]\d*$',"",operator[0])           

            for j in range(len(c_variables)):
                self.coefficient_matrix[c_variables[j]][i] = coefficient_i[j]
            self.coefficient_matrix["b"][i] = b_i

            
            if(operator == "<="):
                coefficient_matrix_columns = self.coefficient_matrix.columns.tolist()
                self.coefficient_matrix.insert(coefficient_matrix_columns.index('b'),'s'+str(i+1),np.zeros([number_constains+1]))
                self.coefficient_matrix['s'+str(i+1)][i] = 1
                self.coefficient_matrix = self.coefficient_matrix.rename(index={i:'s'+str(i+1)})
                     
            if(operator == ">="):
                coefficient_matrix_columns = self.coefficient_matrix.columns.tolist()
                self.coefficient_matrix.insert(coefficient_matrix_columns.index('b'),'s'+str(i+1),np.zeros([number_constains+1]))
                self.coefficient_matrix['s'+str(i+1)][i] = -1
                self.coefficient_matrix = self.coefficient_matrix.rename(index={i:'s'+str(i+1)})

            if(operator == "="):
                coefficient_matrix_columns = self.coefficient_matrix.columns.tolist()
                self.coefficient_matrix.insert(coefficient_matrix_columns.index('b'),'u'+str(i+1),np.zeros([number_constains+1]))
                self.coefficient_matrix['u'+str(i+1)][i] = 1
                self.coefficient_matrix = self.coefficient_matrix.rename(index={i:'u'+str(i+1)})
        
        #建立rc这一行的初始系数
        self.coefficient_matrix = self.coefficient_matrix.rename(index={i+1:"rc"})   
        for variable in d_variables:
            self.coefficient_matrix[variable][i+1] = cost[coefficient_matrix_columns.index(variable)]
        
        #将coefficient_matrix复制一份，作为初始系数表
        self.initial_coefficient_matrix = self.coefficient_matrix.copy()
        print("\nThe  initial coefficient matrix is \n",self.initial_coefficient_matrix)
        
    
    def simplex(self):
        #建立初始可行基
        self.feasible_base = self.coefficient_matrix.index.tolist()
        self.feasible_base.remove('rc')
        
        #显示初始可行基
        print("The initial feasible base is ", self.feasible_base)
        
        while(sum(self.coefficient_matrix.loc['rc']>0)!=0):
            #确定pivot的位置
            pivot_colume = self.coefficient_matrix.loc['rc'].idxmax()
            pivot_row = (self.coefficient_matrix['b'][(self.coefficient_matrix[pivot_colume]>0)&\
            (self.coefficient_matrix.index!='rc')]/self.coefficient_matrix[pivot_colume]\
            [(self.coefficient_matrix[pivot_colume]>0)&(self.coefficient_matrix.index!='rc')]).idxmin()
            print(pivot_row)
            if pd.isnull(pivot_row):
                print("The problem is unbounded!")
                return
            print("pivot location:",pivot_row,pivot_colume,"; pivot is ",self.coefficient_matrix[pivot_colume][pivot_row])
            print("\n================================================\n")
            
            #更新新可行基并显示,同时更新系数矩阵的index
            self.feasible_base[self.feasible_base.index(pivot_row)] = pivot_colume
            self.coefficient_matrix = self.coefficient_matrix.rename(index={pivot_row:pivot_colume})
            print("The current feasible base is ", self.feasible_base)
            
            #完成基的转换之后，将pivot的行和列都设置为pivot_colume
            pivot_row = pivot_colume
            
            #将pivote的值变为1
            self.coefficient_matrix.loc[pivot_row] = self.coefficient_matrix.loc[pivot_row]/self.coefficient_matrix[pivot_colume][pivot_row]
            
            #对每一行进行初等运算,除了pivote自己那行
            for idx in self.coefficient_matrix.index:
                if idx == pivot_row:
                    continue
                else:
                    self.coefficient_matrix.loc[idx] = self.coefficient_matrix.loc[idx]-self.coefficient_matrix[pivot_colume][idx]*self.coefficient_matrix.loc[pivot_row]
            
            #显示新的系数矩阵
            print("The  current coefficient matrix is\n",self.coefficient_matrix)
        
        #迭代结束后求解并显示解
        #获取目标函数的值
        print("\n================================================\n")
        self.objective_value = -self.coefficient_matrix['b']['rc']
        print("The",self.optimal_object,'value of the objective function is ',self.objective_value)
        
        #获得各个决策变量的解
        for dv in list(self.decision_variables.keys()):
            if dv in self.feasible_base:    
                self.decision_variables[dv] = self.coefficient_matrix['b'][dv]
                print(dv,'=%.2f'%self.decision_variables[dv])
            else:
                print(dv,'=',self.decision_variables[dv])
                
        #计算对偶问题的解
        #从cost字典构建cost的np数组
        cost = []
        for base in self.feasible_base:
            if base in list(self.cost.keys()):
                cost.append(self.cost[base])
            else:
                cost.append(0)
        cost = np.array(cost)
        
        #从初始系数矩阵获得基对应的系数
        B = self.initial_coefficient_matrix[self.feasible_base][self.coefficient_matrix.index!='rc']
        self.B = B.values
        B_inverse = np.linalg.inv(B)
        
        #计算对偶问题的解
        self.dual_solution = np.dot(cost,B_inverse)
        
        #显示对偶问题的解
        print("\nThe solution of dual problem is")
        for i in range(len(self.dual_solution)):
            print('y'+str(i+1)+' = %.2f'%self.dual_solution[i])
                
                
    def dual_simplex(self):
        print("对偶单纯形法")
        
    def sensitivity_analysis(self):
        print("敏感性分析")  
        
if __name__ == "__main__":   
    model = PL()
    model.simplex()
