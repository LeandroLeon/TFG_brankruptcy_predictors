import numpy as np


def print_on_file(text, filename='test.txt'):
    file = open(filename, 'a+')
    print(text, file=file)
    file.close()


def format_scores(scores):
    result = 'Accuracy: %.4f (%.4f)' % (np.mean(scores['test_accuracy']),
                                        np.std(scores['test_accuracy']))
    result += '\nPrecision: %.4f (%.4f)' % (np.mean(scores['test_precision']),
                                            np.std(scores['test_precision']))
    result += '\nRecall: %.4f (%.4f)' % (np.mean(scores['test_recall']),
                                         np.std(scores['test_recall']))
    result += '\nSpecificity: %.4f (%.4f)' % (np.mean(scores['test_specificity']),
                                         np.std(scores['test_specificity']))
    result += '\nF1 Score: %.4f (%.4f)' % (np.mean(scores['test_f1_score']),
                                           np.std(scores['test_f1_score']))
    return result


def get_metrics(tn, fp, fn, tp):
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return {'accuracy': accuracy,
            'recall': recall,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score}
