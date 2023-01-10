import matplotlib.pyplot as plt


def create_feature_importances_dataframe(importance, columns):
    # create importances dataframe
    importances = pd.DataFrame(importance, columns=['importances'])
    importances['feature'] = columns
    importances['type of feature'] = ['fundamental', 'fundamental', 'fundamental', 'fundamental', 'fundamental',
                                     'fundamental', 'fundamental', 'fundamental', 'fundamental', 'fundamental',
                                     'fundamental', 'fundamental', 'fundamental', 'fundamental', 'fundamental',
                                     'fundamental', 'fundamental', 'fundamental', 'fundamental', 'technical',
                                     'macro', 'macro', 'macro', 'macro', 'macro', 'macro', 'macro', 'macro', 'macro',
                                     'macro', 'macro', 'fundamental', 'fundamental', 'fundamental']

    # sort importances in ascending order
    importances.sort_values(by='importances', inplace=True)

    return importances

def plot_feature_importances(importances):
    # plot feature importances
    for i in range(len(importances)):
       if importances['type of feature'].iloc[i] == 'macro':
           plt.bar(importances['feature'].iloc[i], importances['importances'].iloc[i], color='b')
       elif importances['type of feature'].iloc[i] == 'technical':
           plt.bar(importances['feature'].iloc[i], importances['importances'].iloc[i], color='r')
       elif importances['type of feature'].iloc[i] == 'fundamental':
           plt.bar(importances['feature'].iloc[i], importances['importances'].iloc[i], color='g')
    plt.xticks(rotation=90)
    plt.tight_layout()
