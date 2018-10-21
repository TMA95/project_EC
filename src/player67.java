import org.vu.contest.ContestEvaluation;
import org.vu.contest.ContestSubmission;

import java.util.Properties;
import java.util.Random;

public class player67 implements ContestSubmission {
    Random rnd;
    ContestEvaluation evaluation;
    private int evaluationLimit;
    private int populationSize;
    private int subPops;
    private int cycle;
    //private EAPopulation population;
    // private Settings stg;


    //settings
    private int dimensionNum = 10;

    private double MUTATION_RATE = 0.1;
    private double MUTATION_STEP_SIZE = 0.05;

    private int subPopulations = 1;
    private static int exchangeRound = 50;
    private int islands = 0;
    private final int NUM_EXCHANGES = 5;

    private String parentSelection = "LINEAR"; //FPS, LINEAR, EXPONENTIAL, RANDOM;
    private String recombination = "WHOLE_ARITHMETIC"; //DISCRETE, SIMPLE_ARITHMETIC, SINGLE_ARITHMETIC, WHOLE_ARITHMETIC, BLEND_RECOMBINATION
    private String mutation = "UNCORRELATED_N"; //UNIFORM, NON_UNIFORM, UNCORRELATED, UNCORRELATED_N, CORRELATED;
    private String survivorSelection = "MU_PLUS_LAMBDA"; //OFFSPRING, MU_PLUS_LAMBDA;

    private boolean crowding = false;
    private boolean fitnessSharing = false;
    private boolean islandModel = false;

    private double tau = 0.025;
    private double tau2 = 0.2;
    private double epsilon = 0.1;


    private  boolean bentCigar = false;
    private  boolean katsuura = false;
    private  boolean schaffers = false;


    public player67() {
        rnd = new Random();
    }

    public void setSeed(long seed) {
        // Set seed of algorithm's random process
        rnd.setSeed(seed);
    }

    public void setEvaluation(ContestEvaluation evaluation) {
        // Set evaluation problem used in the run
        this.evaluation = evaluation;

        // Get evaluation properties
        Properties props = evaluation.getProperties();
        // Get evaluation limit
        evaluationLimit = Integer.parseInt(props.getProperty("Evaluations"));
        // Property keys depend on specific evaluation
        // E.g. double param = Double.parseDouble(props.getProperty("property_name"));
        boolean isMultimodal = Boolean.parseBoolean(props.getProperty("Multimodal"));
        boolean hasStructure = Boolean.parseBoolean(props.getProperty("Regular"));
        boolean isSeparable = Boolean.parseBoolean(props.getProperty("Separable"));

        if (isMultimodal) {
            System.out.println("Function is Multimodal.");
        }
        if (hasStructure) {
            System.out.println("Function has structure.");
        }
        if (isSeparable) {
            System.out.println("Function is separable.");
        }

        bentCigar = !(isMultimodal || hasStructure || isSeparable);
        katsuura = isMultimodal && !(hasStructure || isSeparable);
        schaffers = isMultimodal && hasStructure && !isSeparable;

    }

    public void run() {


        if (bentCigar) {
            populationSize = 40;
            islands = 0;

            tau = 0.025;
            tau2 = 0.2;
            epsilon = 0.1;
        }
        if (katsuura) {
            populationSize = 100;
            islands = 5;

            tau = 0.025;
            tau2 = 0.2;
            epsilon = 0.2;

            if (islands > 1) {
                islandModel = true;
                crowding = false;
                fitnessSharing = false;
                subPopulations = populationSize / islands;
            }
        }

        if (schaffers) {
            populationSize = 50;
            islands = 5;

            tau = 0.025;
            tau2 = 0.2;
            epsilon = 0.2;

            if (islands > 1) {
                islandModel = true;
                crowding = false;
                fitnessSharing = false;
                subPopulations = populationSize / islands;
            }

        }

        System.out.print("Population size: " + populationSize);
        if (islandModel) {
            System.out.print(" (" + subPopulations + " subpopulations of size ");
            System.out.print((populationSize / subPopulations) + ")");
        }
        System.out.println();

        int evals = evaluationLimit;
        //int evals = 2*populationSize;
        double eval_frac = ((double) evals) / evaluationLimit;
        double mutation_epsilon = epsilon;

        // Create initial population and evaluate the fitness
        if (islandModel) {
            //population = new IslandModel(populationSize, new Settings(), rnd);
            islandModel();
        } else {
            Population population = new Population(populationSize, rnd, dimensionNum, MUTATION_STEP_SIZE, MUTATION_RATE, tau, tau2);

            evals -= population.evaluateInitialPopulation(evaluation);

            cycle = 0;
            while (evals > 0)
            {
                // Time dependent variables
                eval_frac = ((double) evals) / evaluationLimit;
                mutation_epsilon = epsilon * Math.pow(eval_frac, 4);

                // Select Parents
                population.selectParents(parentSelection, fitnessSharing);

                // Apply crossover / mutation operators
                population.crossover(crowding, recombination);
                population.mutate(mutation, mutation_epsilon);

                // Check fitness of unknown function
                try {
                    evals -= population.evaluateOffspring(evaluation);
                } catch (NullPointerException e) {
                    System.out.println("\033[1mEvaluation limit reached!\033[0m");
                    break;
                }

                // Select survivors
                population.selectSurvivors(survivorSelection);

                cycle++;
            }

            System.out.println("Evolutionary Cycles: " + cycle);
        }
    }

    public void islandModel() {

        int evals = evaluationLimit;
        //int evals = 2*populationSize;
        double eval_frac = ((double) evals) / evaluationLimit;
        double mutation_epsilon = epsilon;

        Population[] population = new Population[islands];
        String mutationMethod;

        int subPopulationSize = populationSize / islands;


        for (int i = 0; i < islands; i++) {

            population[i] = new Population(populationSize, rnd, dimensionNum, MUTATION_STEP_SIZE, MUTATION_RATE, tau, tau2);
        }

        for (int i = 0; i < islands; i++) {
            evals -= population[i].evaluateInitialPopulation(evaluation);
        }

        cycle = 0;
        while (evals > 0) {
            if (cycle > 0 && cycle % exchangeRound == 0) {
                //excange individuals betwine islands
                int n = NUM_EXCHANGES;
                for (int i = 0; i < islands; i++) {
                    population[i].selectBest(n);
                    population[i].removeWorst(n);
                }

                for (int i = 1; i <= islands; i++) {
                    int neighbour = i;
                    if (i == islands) {
                        neighbour = 0;
                    }
                    population[i - 1].addExchange(population[neighbour].getSelectedForExchange());
                }
            }


            // Time dependent variables
            eval_frac = ((double) evals) / evaluationLimit;
            mutation_epsilon = epsilon * Math.pow(eval_frac, 4);

            // Select Parents
            for (int i = 0; i < islands; i++) {
                population[i].selectParents(parentSelection, fitnessSharing);
            }
            // Apply crossover / mutation operators
            for (int i = 0; i < islands; i++) {
                population[i].crossover(crowding, recombination);
            }
            for (int i = 0; i < islands; i++) {
                population[i].mutate(mutation, mutation_epsilon);
            }
            // Check fitness of unknown function
            try {
                for (int i = 0; i < islands; i++) {
                    evals -= population[i].evaluateOffspring(evaluation);
                }

            } catch (NullPointerException e) {
                System.out.println("\033[1mEvaluation limit reached!\033[0m");
                break;
            }

            // Select survivors
            for (int i = 0; i < islands; i++) {
                population[i].selectSurvivors(survivorSelection);
            }
            cycle++;
        }
    }



}
