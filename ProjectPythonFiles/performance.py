import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

class Evaluator:
    """
    A class for evaluating a biometric system's performance.
    """

    def __init__(self,
                 num_thresholds,
                 genuine_scores,
                 impostor_scores,
                 plot_title,
                 epsilon=1e-12):
        """
        Initialize the Evaluator object.

        Parameters:
        - num_thresholds (int): Number of thresholds to evaluate.
        - genuine_scores (array-like): Genuine scores for evaluation.
        - impostor_scores (array-like): Impostor scores for evaluation.
        - plot_title (str): Title for the evaluation plots.
        - epsilon (float): A small value to prevent division by zero.
        """
        self.num_thresholds = num_thresholds
        self.thresholds = np.linspace(-0.1, 1.1, num_thresholds)
        self.genuine_scores = genuine_scores
        self.impostor_scores = impostor_scores
        self.plot_title = plot_title
        self.epsilon = epsilon
        self.FPR = None 
        self.FNR = None
        self.TPR = None

    def get_dprime(self):
        """
        Calculate the d' (d-prime) metric.

        Returns:
        - float: The calculated d' value.
        """
        genuine_mean = np.mean(self.genuine_scores)
        impostor_mean = np.mean(self.impostor_scores)
        genuine_std = np.std(self.genuine_scores)
        impostor_std = np.std(self.impostor_scores)

        x = np.abs(genuine_mean - impostor_mean)
        y = np.sqrt(0.5 * (genuine_std**2 + impostor_std**2))
        return x / (y + self.epsilon)

    def plot_score_distribution(self):
        """
        Plot the distribution of genuine and impostor scores, including the threshold line.
        """
        plt.figure()

        # Plot the histogram for genuine scores
        plt.hist(
            self.genuine_scores,
            color='green',
            lw=2,
            histtype='step',
            hatch='//',
            label='Genuine Scores'
        )
        plt.yscale('log')
        # Plot the histogram for impostor scores
        plt.hist(
            self.impostor_scores,
            color='red',
            lw=2,
            histtype='step',
            hatch='\\\\',
            label='Impostor Scores'
        )

        eer = self.get_EER(self.FPR, self.FNR)
        threshold_at_eer = self.thresholds[np.argmin(np.abs(np.array(self.FPR) - np.array(self.FNR)))]
        plt.axvline(x=threshold_at_eer, color='black', linestyle='--', linewidth=2, label='Threshold')

        annotation_text = f"Score threshold, t={threshold_at_eer:.2f}, at EER\nFPR={self.FPR[np.argmin(np.abs(np.array(self.FPR) - np.array(self.FNR)))]:.2f}, FNR={self.FNR[np.argmin(np.abs(np.array(self.FPR) - np.array(self.FNR)))]:.2f}"
        plt.text(0.65, 0.75, annotation_text, transform=plt.gca().transAxes,
                 bbox={'facecolor': 'lightgrey', 'alpha': 0.7}) 

        # Set the x-axis limit to ensure the histogram fits within the correct range
        plt.xlim([-0.05, 1.05])

        # Add grid lines for better readability
        plt.grid(color='gray', linestyle='--', linewidth=0.5)

        # Add legend to the upper left corner with a specified font size
        plt.legend(
            loc='upper left',
            fontsize=15
        )

        # Set x and y-axis labels with specified font size and weight
        plt.xlabel(
            'Matching Scores',
            fontsize=15,
            weight='bold'
        )

        plt.ylabel(
           'Score frequency',
            fontsize=15,
            weight='bold'
        )

        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # Set font size for x and y-axis ticks
        plt.xticks(
            fontsize=15
        )

        plt.yticks(
            fontsize=15
        )

        # Add a title to the plot with d-prime value and system title
        plt.title('Score Distribution Plot\nd-prime= %.2f\nSystem %s' %
                  (self.get_dprime(),
                   self.plot_title),
                  fontsize=15,
                  weight='bold')

        plt.tight_layout()
        # Save the figure before displaying it
        plt.savefig('score_distribution_plot_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")

        # Display the plot after saving
        plt.show()

        # Close the figure to free up resources
        plt.close()

        return

    def get_EER(self, FPR, FNR):
            """
            Calculate the Equal Error Rate (EER).

            Parameters:
            - FPR (list or array-like): False Positive Rate values.
            - FNR (list or array-like): False Negative Rate values.

            Returns:
            - float: Equal Error Rate (EER).
            """
            FPR_arr = np.array(FPR)
            FNR_arr = np.array(FNR)
            idx = np.argmin(np.abs(FPR_arr - FNR_arr))
            EER = (FPR_arr[idx] + FNR_arr[idx]) / 2.0
            return EER

    def plot_det_curve(self, FPR, FNR):
        """
        Plot the Detection Error Tradeoff (DET) curve.
        Parameters:
         - FPR (list or array-like): False Positive Rate values.
         - FNR (list or array-like): False Negative Rate values.
        """

        # Calculate the Equal Error Rate (EER) using the get_EER method
        EER = self.get_EER(FPR, FNR)

        # Create a new figure for plotting
        plt.figure()

        # Plot the Detection Error Tradeoff Curve
        plt.plot(
            FPR,
            FNR,
            lw=2,
            color='black'
        )

        # Add a text annotation for the EER point on the curve
        # Plot the diagonal line representing random classification
        # Scatter plot to highlight the EER point on the curve
        plt.text(EER + 0.07, EER + 0.07, "EER", style='italic', fontsize=12,
                        bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        plt.scatter([EER], [EER], c="black", s=100)


        # Set the x and y-axis limits to ensure the plot fits within the range
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

        # Add grid lines for better readability
        plt.grid(color='gray', linestyle='--', linewidth=0.5)

        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # Set x and y-axis labels with specified font size and weight
        plt.xlabel(
            'False Accept Rate',
            fontsize=15,
            weight='bold'
        )

        plt.ylabel(
            'False Reject Rate',
            fontsize=15,
            weight='bold'
        )

        # Add a title to the plot with EER value and system title
        plt.title(
            'Detection Error Tradeoff Curve \nEER = %.5f\nSystem %s' %
            (EER, self.plot_title),
            fontsize=15,
            weight='bold'
        )

        # Set font size for x and y-axis ticks
        plt.xticks(
            fontsize=15
        )

        plt.yticks(
            fontsize=15
        )

        plt.tight_layout()
        # Save the plot as an image file
        plt.savefig(
            'DET_Curve_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight"
        )

        # Display the plot
        plt.show()

        # Close the plot to free up resources
        plt.close()

        return

    def plot_roc_curve(self, FPR, TPR):
        """
        Plot the Receiver Operating Characteristic (ROC) curve.
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - TPR (list or array-like): True Positive Rate values.
        """

        # Create a new figure for the ROC curve
        plt.figure()
        # Plot the ROC curve using FPR and TPR with specified attributes
        plt.plot(FPR, TPR, lw=2, color='black')

        # Set x and y axis limits, add grid, and remove top and right spines
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # Set labels for x and y axes, and add a title

        plt.xlabel('False Accept Rate', fontsize=15, weight='bold')
        plt.ylabel('True Accept Rate', fontsize=15, weight='bold')
        plt.title('Receiver Operating Characteristic Curve\nArea Under Curve = %.5f\nSystem %s' %
                  (metrics.auc(FPR, TPR),
                  self.plot_title),
                  fontsize=15,
                  weight='bold')

        # Set font sizes for ticks, x and y labels
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.tight_layout()

        # Save the plot as a PNG file and display it
        plt.savefig('ROC_Curve_(%s).png' % self.plot_title, dpi=300, bbox_inches='tight')
        plt.show()

        # Close the figure to free up resources
        plt.close()
        return

    def compute_rates(self):
        # Initialize lists for False Positive Rate (FPR), False Negative Rate (FNR), and True Positive Rate (TPR)
        FPR = []
        FNR = []
        TPR = []

        # Iterate through threshold values and calculate TP, FP, TN, and FN for each threshold
        for threshold in self.thresholds:
            TP = 0 
            FP = 0 
            TN = 0
            FN = 0 

            for score in self.genuine_scores:
                if score >= threshold:
                    TP += 1 
                else:
                    FN += 1  

            for score in self.impostor_scores:
                if score >= threshold:
                    FP += 1  
                else:
                    TN += 1 
            # Calculate FPR, FNR, and TPR based on the obtained values
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
            fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
            tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
            # Append calculated rates to their respective lists
            FPR.append(fpr)
            FNR.append(fnr)
            TPR.append(tpr)

        # Return the lists of FPR, FNR, and TPR
        self.FPR = FPR 
        self.FNR = FNR
        self.TPR = TPR
        return FPR, FNR, TPR
def main():
    np.random.seed(1)
    systems = ['A', 'B', 'C']
    for system in systems:
        # Use np.random.random sample() to generate a random float between
        # 0.5 and 0.9 and another random float between 0.0 and 0.2. Use these
        # as the μ (mean) and σ (standard deviation), respectively, to generate
        # 400 genuine scores using np.random.normal().
        genuine_mean = 0.5 + (np.random.random_sample() * (0.9 - 0.5))
        genuine_std = np.random.random_sample() * 0.2
        genuine_scores = np.random.normal(genuine_mean, genuine_std, 400)

        # Repeat with μ ∈ [0.1, 0.5) and σ ∈ [0.0, 0.2) to generate 1,600
        # impostor scores.
        impostor_mean = 0.1 + (np.random.random_sample() * (0.5 - 0.1))
        impostor_std = np.random.random_sample() * 0.2
        impostor_scores = np.random.normal(impostor_mean, impostor_std, 1600)

        # Creating an instance of the Evaluator class
        evaluator = Evaluator(
            epsilon=1e-12,
            num_thresholds=200,
            genuine_scores=genuine_scores,
            impostor_scores=impostor_scores,
            plot_title=f"{system}"
        )

        FPR, FNR, TPR = evaluator.compute_rates()
        evaluator.plot_score_distribution()
        evaluator.plot_det_curve(FPR, FNR)
        evaluator.plot_roc_curve(FPR, TPR)

if __name__ == "__main__":
    main()

