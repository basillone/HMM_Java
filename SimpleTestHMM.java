/**
 * This is a class to test my HMM class on the simple dataset.
 * This runs the file test and then the console test to accept and tag console input from the user.
 *
 * @author Basil Lone
 */

public class SimpleTestHMM {
    public static void main(String[] args) throws Exception {
        // SIMPLE TESTING
        HMM simple_test = new HMM();

        // Model training
        simple_test.trainModel("data/simple-train-tags.txt", "data/simple-train-sentences.txt");

        // Model testing
        simple_test.fileTester("data/simple-test-sentences.txt", "predictions/simple-test-predictions.txt");

        // Get metrics on the performance of the model
        simple_test.getMetrics("data/simple-test-tags.txt", "predictions/simple-test-predictions.txt");

        // Run the console test
        simple_test.consoleTester();
    }
}
