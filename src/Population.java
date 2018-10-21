import org.vu.contest.ContestEvaluation;

import java.util.*;


public class Population
{
    private int size;
    private double offspringRatio;
    private int offspringSize;
    private List<Individual> population;
    private List<Individual> matingPool;
    private List<Individual> offspring;
    private final int N;
    private final int numParents = 2;
    private Random rnd;
    //private Settings stgs;
    private Individual[] exchange;
    private double mutationStepSize;
    private double mutationRate;
    private double t;
    private double t2;


    public Population(int size,  Random rnd, int diemsion,double mutation_step, double mutationRate, double tau, double taut  )
    {
        mutationStepSize =mutation_step;
        this.mutationRate = mutationRate;
        t = tau;
        t2 =taut;

        this.size = size;
        //this.stgs = stgs;
        this.rnd = rnd;
        this.N = diemsion;

        population = new ArrayList<Individual>();
        matingPool = new ArrayList<Individual>();
        offspring = new ArrayList<Individual>();

        offspringRatio = 1.0;
        offspringSize = (int) (size * offspringRatio);

        populate(rnd);
    }

    private void populate(Random rnd)
    {
        for (int i = 0; i < size; i++) {
            double[] candidate = new double[N];
            for (int j = 0; j < N; j++) {
                candidate[j] = rnd.nextDouble() * 5.0;
                if (rnd.nextBoolean()) {
                    candidate[j] *= -1;
                }
            }
            population.add(new Individual(candidate, mutationStepSize,  mutationRate,t,t2));
        }
    }
    public int evaluateInitialPopulation(ContestEvaluation evaluation)
    {
        int evals = 0;
        for (Individual parent: population) {
            parent.fitness = (double) evaluation.evaluate(parent.value);
            evals++;
        }
        return evals; // return number of evaluations performed
    }

    public int evaluateOffspring(ContestEvaluation evaluation)
    {
        int evals = 0;
        for (Individual child: offspring) {
            child.fitness = (double) evaluation.evaluate(child.value);
            evals++;
        }
        return evals; // return number of evaluations performed
    }


    public void selectParents(String method, boolean fitnessSharing )
    {
        switch(method) {
            case "RANDOM":
                for (Individual parent: population) {
                    matingPool.add(parent);
                }
                break;
            case "LINEAR":
            case "EXPONENTIAL":
                rankingSelection(method);
                break;
            case "FPS":
                if (fitnessSharing) {
                    applyFitnessSharing();
                }
                fitnessProportionalSelection();
                break;
        }
    }


    private void fitnessProportionalSelection()
    {
        double sumFitness = 0.0;
        for (Individual ind: population) {
            sumFitness += ind.fitness;
        }
        for (Individual ind: population) {
            ind.probability = ind.fitness / sumFitness;
        }
        sampleParents();
    }


    private void rankingSelection(String method)
    {
        sortPop();
        int rank = population.size() - 1;
        for (int i = 0; i < population.size(); i++) {
            population.get(i).rank = rank-i;
        }

        switch (method) {
            case "LINEAR":
                for (Individual ind: population) {
                    ind.calcualteLinearRankProbability(size);
                }
                break;
            case "EXPONENTIAL":
                double normalisation = 0.0;
                for (Individual ind: population) {
                    ind.calculateExponentialRankProbability();
                    normalisation += ind.probability;
                }
                for (Individual ind: population) {
                    ind.probability /= normalisation;
                }
                break;
        }
        sampleParents();
    }


    private void sampleParents()
    {
        double r = (rnd.nextDouble() / ((double) offspringSize));
        int i = 0;
        double cumProbability = 0.0;
        while (matingPool.size() < offspringSize) {
            cumProbability += population.get(i).probability;

            while (r <= cumProbability) {
                matingPool.add(population.get(i));
                r += 1 / ((double) offspringSize);
            }
            i++;
        }
    }

    /**************************************************************
     * Recombination                                              *
     **************************************************************/
    public void crossover(boolean crowding,String recombinationMethod)
    {
        if(crowding) {
            deterministicCrowding(recombinationMethod);
            return;
        }

        offspring.clear();
        double[][] parents = new double[numParents][N];
        double[][] children;

        for (int i = 0; i < offspringSize; i += numParents) {
            for (int j = 0; j < numParents; j++) {
                int index = rnd.nextInt(matingPool.size());
                parents[j] = matingPool.get(index).value;
                matingPool.remove(index);
            }

            children = recombination(recombinationMethod,parents);

            for (int j = 0; j < numParents; j++) {
                offspring.add(new Individual(children[j],mutationStepSize,mutationRate,t,t2));
            }
        }
    }

    private double[][] recombination(String method,double[][] parents)
    {
        switch (method) {
            case "DISCRETE":
                return discreteRecombination(parents);
            case "SIMPLE_ARITHMETIC":
                return simpleArithmeticRecombination(parents);
            case "SINGLE_ARITHMETIC":
                return singleArithmeticRecombination(parents);
            case "WHOLE_ARITHMETIC":
                return wholeArithmeticRecombination(parents);
            case "BLEND_RECOMBINATION":
                return blendRecombination(parents);
        }
        return parents;
    }

    // Recombination: Discrete Recombination p65
    private double[][] discreteRecombination(double[][] parents)
    {
        double[][] children = new double[numParents][N];
        int parent = 0;
        int split = rnd.nextInt(N-2) + 1; // split should be in interval [1,N-1]
        for (int j = 0; j < N; j++) {
            if (j == split) {
                parent = 1 - parent;
            }
            children[0][j] = parents[parent][j];
            children[1][j] = parents[1-parent][j];
        }
        return children;
    }

    // Recombination: Simple Arithmetic Recombination p65
    private double[][] simpleArithmeticRecombination(double[][] parents)
    {
        double[][] children = new double[numParents][N];
        int parent = 0;
        int k = rnd.nextInt(N-2) + 1; // split should be in interval [1,N-1]
        double alpha = rnd.nextDouble();

        for (int i = 0; i < numParents; i++) {
            for (int j = 0; j < N; j++) {
                if (j < k) {
                    children[i][j] = parents[parent][j];
                } else {
                    children[i][j] = alpha * parents[1-parent][j] + (1 - alpha) * parents[parent][j];
                }
            }
            parent = 1 - parent;
        }

        return children;
    }

    // Recombination: Single Arithmetic Recombination p66
    private double[][] singleArithmeticRecombination(double[][] parents)
    {
        double[][] children = new double[numParents][N];
        int k = rnd.nextInt(N-2) + 1; // split should be in interval [1,N-1]
        double alpha = rnd.nextDouble();

        for (int j = 0; j < N; j++) {
            if (j == (k-1)) {
                children[0][j] = alpha * parents[1][j] + (1 - alpha) * parents[0][j];
                children[1][j] = alpha * parents[0][j] + (1 - alpha) * parents[1][j];
            } else {
                children[0][j] = parents[0][j];
                children[1][j] = parents[1][j];
            }
        }

        return children;
    }

    // Recombination: Whole Arithmetic Recombination p66
    private double[][] wholeArithmeticRecombination(double[][] parents)
    {
        double[][] children = new double[numParents][N];
        double alpha = rnd.nextDouble();

        for (int j = 0; j < N; j++) {
            children[0][j] = alpha * parents[0][j] + (1 - alpha) * parents[1][j];
            children[1][j] = alpha * parents[1][j] + (1 - alpha) * parents[0][j];
        }

        return children;
    }

    // Recombination: Blend Crossover p66
    private double[][] blendRecombination(double[][] parents)
    {
        // Blend Crossover p. 67
        double[][] children = new double[numParents][N];
        double alpha = 0.5;

        for (int p = 0; p < numParents; p++) {
            double u = rnd.nextDouble();
            double gamma = (1 - 2 * alpha) * u - alpha;

            for (int j = 0; j < N; j++) {
                children[p][j] = (1 - gamma) * parents[0][j] + gamma * parents[1][j];

                if (children[p][j] > 5) {
                    children[p][j] = 5;
                } else if (children[p][j] < -5) {
                    children[p][j] = -5;
                }
            }
        }
        return children;
    }

    /**************************************************************
     * Mutation                                                   *
     **************************************************************/
    public void mutate(String mutationMethod, double epsilon)
    {
        for (Individual child: offspring) {
            child.mutate(mutationMethod, epsilon, rnd);
        }
    }

////////
    public void selectSurvivors( String method)
    {
        switch(method) {
            case "OFFSPRING":
                replaceWithOffspring();
                break;
            case "MU_PLUS_LAMBDA":
                mplSel();
                break;
        }
    }

    private void replaceWithOffspring()
    {
        population.clear();
        for (Individual child: offspring) {
            population.add(child);
        }
        offspring.clear();
    }

    // (μ + λ) selection. merge parents and offspring and keep top μ
    private void mplSel()
    {
        int mu = size;
        int lambda = offspringSize;
        population.addAll(offspring);
        sortPop();
        population.subList(mu, mu+lambda).clear(); // Keep the best μ
    }

//

    public void deterministicCrowding(String recombinationMethod)
    {
        double[][] ch;

        double[][] p = new double[numParents][N];
        Collections.shuffle(matingPool);

        for (int i = 0; i < size; i += numParents) {
            p[0] = matingPool.get(i).value;
            p[1] = matingPool.get(i+1).value;

            ch = recombination(recombinationMethod,p);

            for (int j = 0; j < numParents; j++) {
                offspring.add(new Individual(ch[j],mutationStepSize,mutationRate,t,t2));
            }
        }
    }


    private void applyFitnessSharing()
    {
        double sumSharing;

        for (int i = 0; i < size; i++) {
            sumSharing = 0.0;
            for (int j = 0; j < size; j++) {
                sumSharing += sharing(distance(population.get(i), population.get(j)));
            }
            population.get(i).fitness = population.get(i).fitness / sumSharing;
        }
    }

    // Multimodality: Fitness Sharing p92
    private double sharing(double distance)
    {
        int alpha = 1;
        double share = 5.0;
        if (distance <= share) {
            return 1 - Math.pow((distance/share), (double) alpha);
        } else {
            return 0.0;
        }
    }

//////////////////////////////////////////////////
    public void selectBest(int n)
    {
        exchange = new Individual[n];
        sortPop();
        for (int i = 0; i < n; i++) {
            exchange[i] = population.get(i);
        }
    }

    public void selectFromFittestHalf(int n)
    {
        exchange = new Individual[n];
        sortPop();
        for (int i = 0; i < n; i++) {
            int r = rnd.nextInt(size/2);
            exchange[i] = population.get(r);
        }
    }

    public void selectRandom(int n)
    {
        exchange = new Individual[n];
        for (int i = 0; i < n; i++) {
            int r = rnd.nextInt(size);
            exchange[i] = population.get(r);
        }
    }

    public Individual[] getSelectedForExchange()
    {
        Individual[] tmp = new Individual[exchange.length];
        for (int i = 0; i < exchange.length; i++) {
            tmp[i] = new Individual(exchange[i].value,mutationStepSize,mutationRate,t,t2);
        }
        return tmp;
    }

    public void addExchange(Individual[] individuals)
    {
        for (Individual ind: individuals) {
            population.add(ind);
        }
    }


    private void sortPop()
    {
        population.sort(Comparator.comparingDouble(Individual::fit).reversed());
    }


    public void removeWorst(int n)
    {
        sortPop();
        population.subList(size-n, size).clear();
    }


    private double distance(Individual a, Individual b)
    {
        double d = 0.0;
        for (int i = 0; i < a.value.length; i++) {
            d += Math.pow(a.value[i] - b.value[i], 2);
        }
        return Math.sqrt(d);
    }

    public int size()
    {
        return size;
    }


    public void print()
    {
        for (int i = 0; i < size; i++) {
            System.out.println(Arrays.toString(population.get(i).value));
        }
    }

    public void printFitness()
    {
        int i = 0;
        String s = "[";
        for (Individual ind: population) {
            s += (ind.fitness + ", ");
            i++;
            if (i % 8 == 0) {
                s += "\n";
            }
        }
        s += "]\n";
        System.out.print(s);
    }
}
