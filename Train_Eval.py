from BinTech.CNNTrain import *
from BinTech.CNNEval  import *
from BinTech.Utils.ConfigParser import load_config

if __name__=='__main__':
    config_filepath = './Config.py'
    Config=load_config(config_filepath)
    if Config.Training:
        tool=CNNTrain_v1(Config)
        tool.Train()
    else:
        #Eval(Config.Evalue_Model,Config.Evalue_Data_Dir,Config.Batch_Size)
        #Eval_v1(Config.Evalue_Model,\
        #   os.path.join(Config.Evalue_Data_Dir,Config.Evalue_TFRecord_name),\
        #   Config.Image_Size,Config.Class_Num,Config.Batch_Size)
        Eval_v1(Config.Evalue_Model,\
            os.path.join(Config.Evalue_Data_Dir,Config.Evalue_TFRecord_name),\
            Config.Image_Size,Config.Class_Num,Config.Batch_Size)
