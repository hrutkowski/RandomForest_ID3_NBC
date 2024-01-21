from experients_helpers import plot_results

x_val_1 = [[0, 1], [0.5, 0.5], [1, 0]]
x_val_2 = [0.3, 0.6, 0.9]
y_val = [0.98, 0.93, 0.83]
y_val_std = [0.01, 0.03, 0.05]
x_label = 'Stosunek klasyfikatorów'
y_label = 'Dokładność'

plot_results(x_val_1, y_val, y_val_std, x_label, y_label)
plot_results(x_val_2, y_val, y_val_std, x_label, y_label)
