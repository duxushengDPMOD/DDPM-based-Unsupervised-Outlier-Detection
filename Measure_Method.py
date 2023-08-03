import numpy as np
from sklearn.metrics import roc_auc_score
def Measure(output,train_x,label,outlier_number):

    Every_Object_Loss = np.sum(np.square(train_x - output), axis=1)
    Outlier_Factor=Every_Object_Loss


    Sorted_Outlier_Factor = np.argsort(-Outlier_Factor)

    top_n_values = Outlier_Factor[Sorted_Outlier_Factor[:outlier_number]]

    ODA_AbnormalObject_Number = Sorted_Outlier_Factor[:outlier_number]
    ODA_NormalObject_Number=Sorted_Outlier_Factor[outlier_number:]

    Real_NormalObject_Number = np.where(label == 0)[0]
    Real_AbnormalObject_Number = np.where(label == 1)[0]


    TP = len(set(Real_AbnormalObject_Number).intersection(ODA_AbnormalObject_Number))
    FP = len(Real_AbnormalObject_Number)-TP
    TN = len(set(Real_NormalObject_Number).intersection(ODA_NormalObject_Number))
    FN = len(Real_NormalObject_Number) - TN

    AUC = roc_auc_score(label, Outlier_Factor)
    print('AUC= \n',AUC)

    ACC=(TP+TN)/(TP+TN+FP+FN);
    print('ACC= \n',ACC*100)

    DR=TP/(TP+FN);
    print('DR= \n',DR*100)

    P=TP/(TP+FP);
    print('P= \n',P*100)

    FAR=FP/(TN+FP);
    print('FAR= \n',FAR*100)






