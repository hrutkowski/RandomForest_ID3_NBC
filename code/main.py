from scripts_and_experiments.do_experiments import id3_comparison, nbc_comparison, tree_number_influence, \
    classifier_ratio_influence, samples_percentage_influence

id3_comparison('corona')
nbc_comparison('corona')
tree_number_influence('divorce')
classifier_ratio_influence('divorce')
samples_percentage_influence('divorce')