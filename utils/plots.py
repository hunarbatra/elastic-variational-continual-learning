# import matplotlib.pyplot as plt
# import numpy as np

# def plot_results(results_dict, filename, title=''):
#     plt.figure(figsize=(7, 3))
#     for name, results in results_dict.items():
#         # avg_results = []
#         # for i in range(len(results)):
#         #     avg_result = np.mean(results[:i+1])
#         #     avg_results.append(avg_result)
#         plt.plot(np.arange(1, 1+len(results)), results, label=name, marker='o')

#         # Print accuracy values on top of the dots
#         for i, res in enumerate(results):
#             plt.text(i+1, res, f'{res:.2f}', ha='center', va='bottom')

#     plt.xticks(np.arange(1, len(results) + 1))
#     plt.xlabel('Task #')
#     plt.ylabel('Average Accuracy')
#     plt.title(title)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'{filename}.png')
#     plt.show()