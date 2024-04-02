from engine import Engine
import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=('vast', 'pstance', 'covid'), default='vast',
                        help='which dataset to use')
    parser.add_argument('--topic', type=str, choices=('bernie', 'biden', 'trump',
                                                      'bernie,biden', 'bernie,trump',
                                                      'biden,bernie', 'biden,trump',
                                                      'trump,bernie', 'trump,biden',
                                                      'face_masks', 'fauci',
                                                      'stay_at_home_orders', 'school_closures', ''), default='bernie',
                        help='the topic to use')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--l2_reg', type=float, default=5e-5)
    parser.add_argument('--max_grad', type=float, default=0)
    parser.add_argument('--n_layers_freeze', type=int, default=0)
    parser.add_argument('--model', type=str, choices=('bert-base', 'bertweet', 'covid-twitter-bert'),
                        default='bert-base')
    parser.add_argument('--wiki_model', type=str, choices=('', 'bert-base'), default='bert-base')
    parser.add_argument('--n_layers_freeze_wiki', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--inference', type=int, default=1, help='if doing inference or not')

    parser.add_argument('--seed', type=int, default=159) # default=42
    parser.add_argument('--senti_flag', type=bool, default=True, help='use sentiment features or not')
    parser.add_argument('--graph_flag', type=bool, default=True, help='use graph features or not')
    parser.add_argument('--summa_flag', type=bool, default=True, help='use text summarization or not')

    args = parser.parse_args()
    print(args)

    engine = Engine(args)
    y_outputs, y_pred, y_true, y_logits, mask_few_shot, \
    f1_avg, f1_favor, f1_against, f1_neutral, \
    f1_avg_few, f1_favor_few, f1_against_few, f1_neutral_few, \
    f1_avg_zero, f1_favor_zero, f1_against_zero, f1_neutral_zero, = engine.eval(phase='test', get_logits=True)
    
    np.save("data/y_outputs", y_outputs)
    np.save("results/y_pred", y_pred)
    np.save("results/y_true", y_true)
    np.save("results/y_logits", y_logits)
    np.reshape(y_pred, [y_pred.shape(1), 1])
    np.reshape(y_true, [y_true.shape(1), 1])
    raw_res = np.concatenate((y_logits, y_pred, y_true), axis=1)
    print(raw_res)
    print(len(raw_res))
    print(f'Test F1: {f1_avg:.3f}\tTest F1_Favor: {f1_favor:.3f}\t'
        f'Test F1_Against: {f1_against:.3f}\tTest F1_Neutral: {f1_neutral:.3f}\n'
        f'Test F1_Few: {f1_avg_few:.3f}\tTest F1_Favor_Few: {f1_favor_few:.3f}\t'
        f'Test F1_Against_Few: {f1_against_few:.3f}\tTest F1_Neutral_Few: {f1_neutral_few:.3f}\n'
        f'Test F1_Zero: {f1_avg_zero:.3f}\tTest F1_Favor_Zero: {f1_favor_zero:.3f}\t'
        f'Test F1_Against_Zero: {f1_against_zero:.3f}\tTest F1_Neutral_Zero: {f1_neutral_zero:.3f}')