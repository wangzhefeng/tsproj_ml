# -*- coding: utf-8 -*-

# ***************************************************
# * File        : args_tools.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-13
# * Version     : 1.0.011322
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from pathlib import Path

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class DotDict(dict):

    """
    dot.notation access to dictionary attributes
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


def print_args_ts(args):
    # ------------------------------
    # Basic Config
    # ------------------------------
    logger.info(f'{100 * "="}')
    logger.info(f'Args in experiment:')
    logger.info(f'{100 * "-"}')
    logger.info("\033[1m" + "Basic Config" + "\033[0m")
    logger.info(f'  {"Task Name:":<25}{args.task_name:<25}{"Des:":<25}{args.des:<25}')
    logger.info(f'  {"Is Training:":<25}{args.is_training:<25}{"Is Testing:":<25}{args.is_testing:<25}')
    logger.info(f'  {"Is Forecasting:":<25}{args.is_forecasting:<25}')
    logger.info(f'  {"Model ID:":<25}{args.model_id:<25}{"Model:":<25}{args.model:<25}')
    logger.info("")
    # ------------------------------
    # Data Loader
    # ------------------------------
    logger.info("\033[1m" + "Data Loader" + "\033[0m")
    logger.info(f'  {"Data:":<25}{args.data:<25}{"Root Path:":<25}{args.root_path:<25}')
    logger.info(f'  {"Data Path:":<25}{args.data_path:<25}{"Features:":<25}{args.features:<25}')
    logger.info(f'  {"Target:":<25}{args.target:<25}{"Freq:":<25}{args.freq:<25}')
    logger.info(f'  {"Checkpoints:":<25}{args.checkpoints:<25}')
    logger.info(f'  {"Test results:":<25}{args.test_results:<25}')
    logger.info(f'  {"Predict results:":<25}{args.predict_results:<25}')
    logger.info("")
    # ------------------------------
    # task
    # ------------------------------
    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        logger.info("\033[1m" + "Forecasting Task" + "\033[0m")
        logger.info(f'  {"Seq Len:":<25}{args.seq_len:<25}{"Label Len:":<25}{args.label_len:<25}')
        logger.info(f'  {"Pred Len:":<25}{args.pred_len:<25}{"Seasonal Patterns:":<25}{args.seasonal_patterns:<25}')
        logger.info(f'  {"Train ratio:":<25}{args.train_ratio:<25}{"Test ratio:":<25}{args.test_ratio:<25}')
        logger.info(f'  {"Inverse:":<25}{args.inverse:<25}{"Scale:":<25}{args.scale:<25}')
        logger.info(f'  {"Embed:":<25}{args.embed:<25}')
        logger.info("")
    elif args.task_name == 'imputation':
        logger.info("\033[1m" + "Imputation Task" + "\033[0m")
        logger.info(f'  {"Mask Rate:":<25}{args.mask_rate:<25}')
        logger.info("")
    elif args.task_name == 'anomaly_detection':
        logger.info("\033[1m" + "Anomaly Detection Task" + "\033[0m")
        logger.info(f'  {"Anomaly Ratio:":<25}{args.anomaly_ratio:<25}')
        logger.info("")
    # ------------------------------
    # MOdel Parameters
    # ------------------------------
    logger.info("\033[1m" + "Model Parameters" + "\033[0m")
    logger.info(f'  {"Top k:":<25}{args.top_k:<25}{"Num Kernels:":<25}{args.num_kernels:<25}')
    logger.info(f'  {"Enc In:":<25}{args.enc_in:<25}{"Dec In:":<25}{args.dec_in:<25}')
    logger.info(f'  {"C Out:":<25}{args.c_out:<25}{"d model:":<25}{args.d_model:<25}')
    logger.info(f'  {"n heads:":<25}{args.n_heads:<25}{"e layers:":<25}{args.e_layers:<25}')
    logger.info(f'  {"d layers:":<25}{args.d_layers:<25}{"d FF:":<25}{args.d_ff:<25}')
    logger.info(f'  {"Moving Avg:":<25}{args.moving_avg:<25}{"Factor:":<25}{args.factor:<25}')
    logger.info(f'  {"Distil:":<25}{args.distil:<25}{"Dropout:":<25}{args.dropout:<25}')
    logger.info(f'  {"Embed:":<25}{args.embed:<25}{"Activation:":<25}{args.activation:<25}')
    logger.info(f'  {"Output Attention:":<25}{args.output_attention:<25}')
    logger.info("")
    # ------------------------------
    # Run Parameters
    # ------------------------------
    logger.info("\033[1m" + "Run Parameters" + "\033[0m")
    logger.info(f'  {"Num Workers:":<25}{args.num_workers:<25}{"Itr:":<25}{args.itr:<25}')
    logger.info(f'  {"Train Epochs:":<25}{args.train_epochs:<25}{"Batch Size:":<25}{args.batch_size:<25}')
    logger.info(f'  {"Patience:":<25}{args.patience:<25}{"Learning Rate:":<25}{args.learning_rate:<25}')
    logger.info(f'  {"Loss:":<25}{args.loss:<25}{"Lradj:":<25}{args.lradj:<25}')
    logger.info(f'  {"Use Amp:":<25}{args.use_amp:<25}{"Use DTW:":<25}{args.use_dtw:<25}')
    logger.info("")
    # ------------------------------
    # GPU
    # ------------------------------
    logger.info("\033[1m" + "GPU" + "\033[0m")
    logger.info(f'  {"Use GPU:":<25}{args.use_gpu:<25}{"GPU Type:":<25}{args.gpu_type:<25}')
    logger.info(f'  {"Use Multi GPU:":<25}{args.use_multi_gpu:<25}{"Devices:":<25}{args.devices:<25}')
    logger.info("")
    # ------------------------------
    # De-stationary Projector Params
    # ------------------------------
    logger.info("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
    logger.info(f'  {"P Hidden Dims:":<25}{p_hidden_dims_str:<25}{"P Hidden Layers:":<25}{args.p_hidden_layers:<25}') 
    # logger.info("")
    logger.info(f'{100 * "="}')


def print_args_llm(args):
    # ------------------------------
    # Basic Config
    # ------------------------------
    logger.info(f'{100 * "="}')
    logger.info(f'Args in experiment:')
    logger.info(f'{100 * "-"}')
    logger.info("\033[1m" + "Basic Config" + "\033[0m")
    logger.info(f'  {"Task Name:":<25}{args.task_name:<25}{"Task Desc:":<25}{args.des:<25}')
    logger.info(f'  {"Training:":<25}{args.is_train:<25}{"Testing:":<25}{args.is_test:<25}')
    logger.info(f'  {"Inference:":<25}{args.is_inference:<25}')
    logger.info("")
    # ------------------------------
    # Data Loader
    # ------------------------------
    logger.info("\033[1m" + "Data" + "\033[0m")
    logger.info(f'  {"Data Source:":<25}{args.data_source:<25}')
    logger.info(f'  {"Data URL:":<25}{args.data_url:<25}')
    logger.info(f'  {"Data Path:":<25}{args.data_path:<25}')
    logger.info(f'  {"Data File:":<25}{args.data_file:<25}')
    logger.info(f'  {"Data Name:":<25}{args.data_name:<25}{"Train Ratio:":<25}{args.train_ratio:<25}')
    logger.info(f'  {"Batch Size:":<25}{args.batch_size:<25}{"Number Workders:":<25}{args.num_workers:<25}') 
    logger.info("")
    # ------------------------------
    # Model Parameters
    # ------------------------------
    logger.info("\033[1m" + "Model" + "\033[0m")
    logger.info(f'  {"Tokenizer model:":20}{args.tokenizer_model:<25}{"Vocab Size:":<25}{args.vocab_size:<25}')
    logger.info(f'  {"Model:":<25}{args.model_name:<25}{"Context Lenght:":<25}{args.context_length:<25}')
    logger.info(f'  {"Embedding Dim:":<25}{args.embed_dim:<25}{"Dim Feed Forward:":<25}{args.d_ff:<25}')
    logger.info(f'  {"Number Heads:":<25}{args.n_heads:<25}{"Number Layers:":<25}{args.n_layers:<25}')
    logger.info(f'  {"Dropout rate:":<25}{args.dropout:<25}{"QKV Bias:":<25}{args.qkv_bias:<25}')
    logger.info(f'  {"Dtype:":<25}{str(args.dtype):<25}{"Use Amp:":<25}{args.use_amp:<25}')
    logger.info(f'  {"Learning Rate:":<25}{args.learning_rate:<25}{"Initial Learning Rate:":<25}{args.initial_lr:<25}')
    logger.info(f'  {"Min Learning Rate:":<25}{args.min_lr:<25}{"Weight Decay:":<25}{args.weight_decay:<25}')
    logger.info(f'  {"Learning Rate Adjust:":<25}{args.lradj:<25}')
    logger.info("")
    # ------------------------------
    # Run Parameters
    # ------------------------------
    logger.info("\033[1m" + "Training" + "\033[0m")
    logger.info(f'  {"Train Iter:":<25}{args.itrs:<25}{"Train Epochs:":<25}{args.train_epochs:<25}')
    logger.info(f'  {"Seed:":<25}{args.seed:<25}{"Patience:":<25}{args.patience:<25}')
    logger.info(f'  {"Checkpoints:":<25}{args.checkpoints:<25}')
    logger.info(f'  {"Test results:":<25}{args.test_results:<25}')
    logger.info(f'  {"Max New Tokens:":<25}{args.max_new_tokens:<25}{"Use Cache:":<25}{args.use_cache:<25}')
    logger.info("")
    # ------------------------------
    # Device Parameters
    # ------------------------------
    logger.info("\033[1m" + "Device" + "\033[0m")
    logger.info(f'  {"Use GPU:":<25}{args.use_gpu:<25}{"GPU Type:":<25}{args.gpu_type:<25}')
    logger.info(f'  {"Use DP:":<25}{args.use_dp:<25}{"Use DDP:":<25}{args.use_ddp:<25}')
    logger.info("")
    # logger.info("")
    logger.info(f'{100 * "="}')




# 测试代码 main 函数
def main():
    dct = {
        'scalar_value': 1, 
        'nested_dict': {
            'value': 2, 
            'nested_nested': {
                'x': 21
            }
        }
    }
    dct = DotDict(dct)

    print(dct.nested_dict.nested_nested.x)

if __name__ == "__main__":
    main()
