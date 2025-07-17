/**
 * This is a class to test my HMM class on the Brown corpus dataset.
 * This runs the file test and then the console test to accept and tag console input from the user.
 *
 * @author Basil Lone
 */

public class BrownTestHMM {
    public static void main(String[] args) throws Exception {
        // BROWN TESTING
        HMM brown_test = new HMM();

        // Model training
        brown_test.trainModel("data/brown-train-tags.txt", "data/brown-train-sentences.txt");

        // Model testing
        brown_test.fileTester("data/brown-test-sentences.txt", "predictions/brown-test-predictions.txt");

        // Get metrics on the performance of the model
        brown_test.getMetrics("data/brown-test-tags.txt", "predictions/brown-test-predictions.txt");

        // Run the console test
        brown_test.consoleTester();
    }
}
