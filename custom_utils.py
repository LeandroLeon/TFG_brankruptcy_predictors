import numpy as np


def print_on_file(text, filename='results.txt'):
    file = open(filename, 'a+')
    print(text, file=file)
    file.close()


def format_scores(scores):
    result = 'Accuracy: %.4f (%.4f)' % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy']))
    result += '\nPrecision: %.4f (%.4f)' % (np.mean(scores['test_precision_weighted']),
                                            np.std(scores['test_precision_weighted']))
    result += '\nRecall: %.4f (%.4f)' % (np.mean(scores['test_recall_weighted']),
                                         np.std(scores['test_recall_weighted']))
    result += '\nF1 Score: %.4f (%.4f)' % (np.mean(scores['test_f1_weighted']),
                                         np.std(scores['test_f1_weighted']))
    return result
