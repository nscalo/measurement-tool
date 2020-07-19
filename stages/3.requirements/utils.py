def convert_perf_time(perf_text):
    import re
    result = re.split("\n", perf_text)
    order = ['batch_size', 'time_per_batch', 'samples_per_second', 'samples']

    def extract(r):
        l = r.split(" ")
        return l[len(l)-1]

    result = list(map(lambda x: float(x), 
    list(filter(lambda x: x.strip() != "", list(map(extract, result))))))

    return [dict(zip(order, result[idx:idx+4])) for idx in range(0,len(result),4)]

def convert_accuracy_time(accuracy_text):
    import re
    result = re.split("\n", accuracy_text)
    order = ['batch_size', 'accuracy']

    def extract(r):
        l = r.split(" ")
        return l[len(l)-1]

    result = list(map(lambda x: float(x), 
    list(filter(lambda x: x.strip() != "", list(map(extract, result))))))

    return [dict(zip(order, result[idx:idx+2])) for idx in range(0,len(result),2)]

def convert_box_predictions_time(box_predictions_text):
    import re
    result = re.split("\n", box_predictions_text)
    prediction = ['prediction_map', 'prediction_map_large', 
    'prediction_map_medium', 'prediction_map_small', 'prediction_map_50iou', 
    'prediction_map_75iou']#
    recall = ['recall_ar1', 'recall_ar10', 
    'recall_ar100', 'recall_ar100_large', 'recall_ar100_medium', 
    'recall_ar100_small']

    result = prediction + recall

    def extract(r):
        l = r.split(" ")
        return l[len(l)-1]

    result = list(map(lambda x: float(x), 
    list(filter(lambda x: x.strip() != "", list(map(extract, result))))))

    return [dict(zip(order, result[idx:idx+12])) for idx in range(0,len(result),12)]