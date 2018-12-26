
# coding: utf-8

import argparse
import customize_flore

#Command Line Arguments

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input', type = str,help='input the image path')
parser.add_argument('checkpoint', help='the model that to be used')
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--category_names',dest='category_names', action='store', default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true',dest='gpu')
torch.cuda.is_available()

pa = parser.parse_args()

path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
path = pa.checkpoint


training_loader, testing_loader, validation_loader = customize_flore.load_data()


customize_flore.load_checkpoint(path)


with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)


probabilities = customize_flore.predict(path_image, model, number_of_outputs, power)


labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])


i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1

print("Done prediction!")
