import java.io.*;
import java.util.*;

/**
 * This is a class to represent a Hidden Markov Model that can be trained to recognize POS (parts of speech).
 *
 * @author Basil Lone
 */

public class HMM {
    private Map<String, Map<String, Double>> obser, trans;    // Maps to store information about the patterns of the speech from the training files
    private final double penalty = -100;                      // The penalty value to assign to a probability when a word is labelled with a tag it has not been observed under
    private final String starting_chr = "#";                  // The character we will use to denote the start of a sentence

    /**
     * Constructs a new instance of the HMM class and create new hashmaps to store POS data from training files
     */
    public HMM () {
        obser = new HashMap<>();
        trans = new HashMap<>();
    }

    /**
     * This reads two files containing sentences and the corresponding tags for each word in the sentences. These files
     * are being used to train the model.
     *
     * @param tagFile       File containing training tags
     * @param wordFile      File containing training words in sentences
     * @throws Exception    This throws an exception if there is a problem with handling the files
     */
    private void fileDataReader (String tagFile, String wordFile) throws Exception {
        // Initialize our text readers for the training files
        BufferedReader inTag;
        BufferedReader inWord;

        // Try to assign our text reader to a file reader for the file with the tag data
        try {
            inTag = new BufferedReader(new FileReader(tagFile));
        }
        // Catch an exception if the file is not found and return to end execution
        catch (FileNotFoundException e) {
            System.err.println("Tag file not found at specified location.\n" + e.getMessage());
            return;
        }

        // Try to assign our text reader to a file reader for the file with the word data
        try {
            inWord = new BufferedReader(new FileReader(wordFile));
        }
        // Catch an exception if the file is not found and return to end execution
        catch (FileNotFoundException e) {
            inTag.close();
            System.err.println("Word file not found at specified location.\n" + e.getMessage());
            return;
        }

        // Try to read the training files and store the data in hash maps
        try {
            String wln;        // String to store a line from the file of words
            String tln;        // String to store a line from the file of tags
            double freq;       // Double to make sure frequencies of transitions and observations are recorded correctly

            // While not at the end of the tag file
            while ((tln = inTag.readLine()) != null) {
                // Split the tags on the current line into separate strings
                String[] tags = tln.toLowerCase().split(" ");

                // Read a line of the word file and split the words into separate strings
                wln = inWord.readLine().toLowerCase();
                String[] words = wln.split(" ");

                // Set our starting character as a back-pointer, so we can build the maps
                String prev = starting_chr;

                // For every tag in the current line
                for (int i = 0; i < tags.length; i++) {

                    // If we have a record of the current tag in the observations map
                    if (obser.containsKey(tags[i])) {
                        // Get its frequency for the observation and update it by 1
                        freq = obser.get(tags[i]).getOrDefault(words[i], 0.0);
                        obser.get(tags[i]).put(words[i], freq+1);
                    }
                    else {
                        // Create a new map for the current tag and set the observation frequency to 1
                        obser.put(tags[i], new HashMap<>());
                        obser.get(tags[i]).put(words[i], 1.0);
                    }

                    // If we have a record of the current tag in the transitions map
                    if (trans.containsKey(prev)) {
                        // Get its frequency for the transition and update it by 1
                        freq = trans.get(prev).getOrDefault(tags[i], 0.0);
                        trans.get(prev).put(tags[i], freq+1);
                    }
                    else {
                        // Create a new map for the current tag and set the transition frequency to 1
                        trans.put(prev, new HashMap<>());
                        trans.get(prev).put(tags[i], 1.0);
                    }

                    // Update the back-pointer string
                    prev = tags[i];
                }
            }
        }
        //  Catch an IOException if there is a problem with reading the files
        catch (IOException e) {
            System.err.println("Error when reading file.\n" + e.getMessage());
        }

        // Try to close the files
        try {
            inTag.close();
            inWord.close();
        }
        // Catch an IOException if there is a problem when closing the files
        catch (IOException e) {
            System.err.println("Cannot close file.\n" + e.getMessage());
        }
    }

    /**
     * This calculates probabilities and then uses those values to get log values to use as scores in our map
     */
    private void probLog () {
        // For every type of transition observed
        for (Map<String, Double> m : trans.values()) {
            // Initialize an integer to keep track of the total number of transitions from a given tag
            int sumF = 0;

            // For every type of transition from the tag, add the frequency of the transition to the sum total
            for (Double d : m.values()) {
                sumF += d;
            }

            // Replace all the frequencies in the transition map with log score probabilities
            for (String tag : m.keySet()) {
                m.put(tag, Math.log(m.get(tag)/sumF));
            }
        }

        // For every tag observed
        for (Map<String, Double> m : obser.values()) {
            // Initialize an integer to keep track of the total number of observations of a given tag
            int sumF = 0;

            // For every type of observation of the tag, add the frequency of the observation to the sum total
            for (Double d : m.values()) {
                sumF += d;
            }

            // Replace all the frequencies in the observation map with log score probabilities
            for (String tag : m.keySet()) {
                m.put(tag, Math.log(m.get(tag)/sumF));
            }
        }
    }

    /**
     * This takes training files and trains the hidden markov model to recognize parts of speech. The probLog method
     * is then called to replace frequencies in the model with log score probabilities.
     *
     * @param tagFile       The training file containing tags
     * @param wordFile      The training file containing words
     */
    public void trainModel (String tagFile, String wordFile) {
        // Try to train the model and assign appropriate log score probabilities
        try {
            fileDataReader(tagFile, wordFile);
            probLog();
        }
        // Catch an exception if there is a problem with file handling
        catch (Exception e) {
            System.err.println("Error in file handling when training.\n" + e.getMessage());
        }
    }

    /**
     * This runs the viterbi algorithm on an array of words forming a sentence to assign each word a tag indicating its
     * part of speech.
     *
     * @param words     An array of words forming a sentence
     * @return          An array of strings containing tags for each word in the sentence
     */
    private String[] viterbi (String[] words) {
        // These will be used to keep track of the current states and their log score probabilities
        Set<String> currStates = new HashSet<>();
        Map<String, Double> currScores = new HashMap<>();

        // This will be used to keep track of the next possible states and their lowest possible log score probabilities
        Map<String, Double> nextScores = new HashMap<>();

        // This will be used for back tracing to find the most likely tags for each word in the sentence
        List<Map<String, String>> tagPaths = new ArrayList<>();

        // Add the starting character and its score to the data structures keeping track of the current possible states
        currStates.add(starting_chr);
        currScores.put(starting_chr, 0.0);

        // Iterate the same number of times as the number of words in the sentence
        for (int i = 0; i < words.length; i++) {

            // This will be used to keep track of the next possible states
            Set<String> nextStates = new HashSet<>();
            // Create new maps for the scores for the next possible states
            nextScores = new HashMap<>();
            tagPaths.add(new HashMap<>());

            // For every possible current state
            for (String cur : currStates) {

                // If there is an existing entry in the transition map
                if (trans.get(cur) != null) {

                    // For every possible transition to a given next state
                    for (String next : trans.get(cur).keySet()) {

                        // Add the state to the set of next possible states
                        nextStates.add(next);

                        // Get the log probability score for the transition
                        Double obsScore = obser.get(next).getOrDefault(words[i], penalty);
                        Double nextScore = currScores.get(cur) + trans.get(cur).get(next) + obsScore;

                        // If we do not have a score for the current possible next state or our new score is greater than the previously recorded one
                        if ((!(nextScores.containsKey(next))) || (nextScores.get(next) < nextScore)) {
                            // Add the next state with its score to the map of next possible states
                            nextScores.put(next, nextScore);
                            tagPaths.get(tagPaths.size() - 1).put(next, cur);
                        }
                    }
                }
            }
            // Update the data structures relating to the current state
            currStates = nextStates;
            currScores = nextScores;
        }

        // Get the best possible final state
        Double max = Double.NEGATIVE_INFINITY;
        String best = new String();
        for (String s : nextScores.keySet()) {
            if (max < nextScores.get(s)) {
                max = nextScores.get(s);
                best = s;
            }
        }

        // Create an array to store the tags
        String[] tags = new String[words.length];

        // Add all the tags on the best possible tag path for the sentence
        for (int i = 0; i < words.length; i++) {
            tags[words.length - 1 - i] = best;
            best = tagPaths.get(words.length - 1 - i).get(best);
        }

        // Return the array of tags
        return tags;
    }

    /**
     * This takes a file of sentences with words and punctuation separated by single spaces. Each sentence must be on
     * a single line. The corresponding tags for words are generated using the viterbi method and written to a
     * separate file.
     *
     * @param testfile      The file containing sentences to use to generate tags
     * @param tagfile       The file to write the generated tags to
     * @throws Exception    This throws an exception if there is a problem with handling the files
     */
    public void fileTester (String testfile, String tagfile) throws Exception {
        // Set up the reader and writer to read the sentences file and write the tags file
        BufferedWriter output;
        BufferedReader input;

        // Try to assign our text writer to a file writer to create a file to write tags to
        try {
            output = new BufferedWriter(new FileWriter(tagfile));
        }
        // Catch an exception if it is not possible to write to the specified location and return to end execution
        catch (IOException e) {
            System.err.println("Can't write to specified location.\n" + e.getMessage());
            return;
        }

        // Try to assign our text reader to a file reader for the file with the sentences
        try {
            input = new BufferedReader(new FileReader(testfile));
        }
        // Catch an exception if the file is not found and return to end execution
        catch (FileNotFoundException e) {
            output.close();
            System.err.println("Word file not found at specified location.\n" + e.getMessage());
            return;
        }

        // Try to read the sentences file, generate the tags, and write the tags to a new file
        try {
            String wln;         // This will store a line of text from the sentences file
            String[] tags;      // This will store the tags corresponding to each word in the line of text

            // While we have not reached the end of the sentences file
            while ((wln = input.readLine()) != null) {
                // Store each word in the current line from the file separately
                String[] words = wln.toLowerCase().split(" ");

                // Run the viterbi method to get a string array with the tags for each word
                tags = viterbi(words);

                // For every tag generated, write the tag to the file
                for (int i = 0; i < tags.length; i++) {
                    output.write(tags[i].toUpperCase());
                    if (i != tags.length - 1) output.write(" ");
                }

                // Write a new line character to separate lines of tags
                output.write("\n");
            }
        }
        //  Catch an IOException if there is a problem with reading or writing the files
        catch (IOException e) {
            System.err.println("Error when reading file.\n" + e.getMessage());
        }

        // Try to close the files
        try {
            input.close();
            output.close();
        }
        // Catch an IOException if there is a problem when closing the files
        catch (IOException e) {
            System.err.println("Cannot close file.\n" + e.getMessage());
        }

    }

    /**
     * This takes a file containing correct tags and generate tags and reports the accuracy of the model in
     * predicting tags.
     *
     * @param tagFile           The file containing correct tags to compare
     * @param predictedFile     The file containing the tags predicted by the model
     * @throws Exception        This throws an exception if there is a problem with handling the files
     */
    public void getMetrics (String tagFile, String predictedFile) throws Exception {
        // Create a map structure to store the wrong predictions and their frequencies
        Map<String, Map<String, Integer>> wrong = new HashMap<>();

        // Integers to keep track of the total number of predictions and incorrect predictions
        int total = 0;
        int wr_count = 0;

        // Initialize our text readers
        BufferedReader inTag;
        BufferedReader inPred;

        // Try to assign our text reader to a file reader for the file with the correct tag data
        try {
            inTag = new BufferedReader(new FileReader(tagFile));
        }
        // Catch an exception if the file is not found and return to end execution
        catch (FileNotFoundException e) {
            System.err.println("Tag file not found at specified location.\n" + e.getMessage());
            return;
        }

        // Try to assign our text reader to a file reader for the file with the predicted tag data
        try {
            inPred = new BufferedReader(new FileReader(predictedFile));
        }
        // Catch an exception if the file is not found and return to end execution
        catch (FileNotFoundException e) {
            inTag.close();
            System.err.println("Word file not found at specified location.\n" + e.getMessage());
            return;
        }

        // Try to read the files and store the data in hash maps
        try {
            // Strings to store single lines from the prediction and correct tag files
            String pln;
            String tln;

            // While we have not reached the end of the file with correct tags
            while ((tln = inTag.readLine()) != null) {
                // Store tags separately as array elements
                String[] tags = tln.split(" ");

                // Read a line of the predicted tags file and store the predicted tags separately as array elements
                pln = inPred.readLine();
                String[] preds = pln.split(" ");

                // For every tag in the line
                for (int i = 0; i < tags.length; i++) {

                    // If the prediction is incorrect
                    if (!(tags[i].equals(preds[i]))) {
                        // If checking a new type of tag, add a new map to wrong
                        if (!(wrong.containsKey(tags[i]))) wrong.put(tags[i], new HashMap<>());

                        // Get the frequency value to increase
                        int freq = wrong.get(tags[i]).getOrDefault(preds[i], 0);

                        // Update the frequency for the tag and increase the count of wrong predictions
                        wrong.get(tags[i]).put(preds[i], freq+1);
                        wr_count++;
                    }

                    // Increase the total count
                    total++;
                }
            }
        }
        //  Catch an IOException if there is a problem with reading the files
        catch (IOException e) {
            System.err.println("Error when reading file.\n" + e.getMessage());
        }

        // Try to close the files
        try {
            inTag.close();
            inPred.close();
        }
        // Catch an IOException if there is a problem when closing the files
        catch (IOException e) {
            System.err.println("Cannot close file.\n" + e.getMessage());
        }

        // Output the metrics for the user
        System.out.println();
        System.out.println("____________________________________________________________________________________");
        System.out.println();
        System.out.println("Total wrong:             " + wr_count);
        System.out.println("Out of:                  " + total);
        System.out.println("Percentage accuracy:     " + ((double) (total - wr_count) / total) * 100);
        System.out.println();
        System.out.println();
        System.out.println("Incorrectly identified: ");
        System.out.println();
        for (String s : wrong.keySet()) {
            String space = " ".repeat(7 - s.length());
            System.out.print(s + space + "predicted as  ");

            for (String t : wrong.get(s).keySet()) {
                String out = t + ":" + wrong.get(s).get(t);
                String spc = " ".repeat(8 - out.length());
                System.out.print(out + spc);
            }
            System.out.println();
        }
        System.out.println();
        System.out.println("____________________________________________________________________________________");
        System.out.println();
    }

    /**
     * This takes user input from the console and outputs predicted tags based on the data used to train the model.
     *
     * @throws Exception        This throws an exception is there is a problem with reading console input
     */
    public void consoleTester () throws Exception {
        // Create a reader to get console input
        BufferedReader input = new BufferedReader(new InputStreamReader(System.in));

        // Loop until the user decides to quit
        while (true) {
            // Output instruction
            System.out.println("Enter q to quit the console input test.");
            System.out.println("Enter a sentence to test > ");

            // Read and format a line of input
            String sentence = input.readLine().strip();

            // If the user has chosen to quit the program, break the loop
            if (sentence.equals("q")) break;

            // Generate tags for the words in the input line
            String[] tags = viterbi(sentence.split(" "));

            // Output all the tags
            for (String s : tags) {
                System.out.print(s.toUpperCase() + " ");
            }

            // New line characters to space out console entries
            System.out.println("\n\n");
        }
    }
}
