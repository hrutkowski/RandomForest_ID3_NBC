from scripts_and_experiments.do_experiments import id3_comparison, nbc_comparison, tree_number_influence, \
    classifier_ratio_influence, samples_percentage_influence

nbc_comparison('loan_approval')
nbc_comparison('corona')
id3_comparison('loan_approval')
id3_comparison('corona')
tree_number_influence('glass')
tree_number_influence('letter')
samples_percentage_influence('glass')
samples_percentage_influence('corona')
classifier_ratio_influence('loan_approval')
classifier_ratio_influence('letter')