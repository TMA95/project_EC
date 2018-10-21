import java.util.Random;


public class Individual
{
    public double[] value;
    public double fitness;
    public double probability;
    public int rank;
    private double[] sigma;
    private double[] alpha;
    private double[][] cov;
    private final double UB = 5.0;
    private final double LB = -UB;
    private double mutationStepSize;
    private double mutationRate;
    private double t = 0;
    private double t2 = 0;

    public Individual(double[] value,double mutation_step, double mutationRate,double tau, double tau2)
    {
        t =tau;
        t2 =tau2;
        mutationStepSize =mutation_step;
        this.mutationRate = mutationRate;
        this.value = value;
        fitness = 0.0;
        probability = 0.0;
        rank = 0;

        sigma = new double[value.length];
        for (int i = 0; i < value.length; i++) {
            sigma[i] = mutationStepSize;
        }
    }

    public double fit()
    {
        return this.fitness;
    }

    public void mutate(String method, double epsilon, Random rnd)
    {
        switch(method) {
            case "UNIFORM":
                uniformMutation(rnd);
                break;
            case "NON_UNIFORM":
                nonUniformMutation(rnd);
                break;
            case "UNCORRELATED":
                uncorMutat(epsilon, rnd);
                break;
            case "UNCORRELATED_N":
                uncorMutWithNStepSizes(epsilon, rnd,t,t2);
                break;
            case "CORRELATED":
                correlatedMutation(epsilon, rnd);
                break;
        }
    }

    public void calcualteLinearRankProbability(int size)
    {
        double s = 2.0;
        this.probability = ((2 - s) / size) + ((2*rank*(s-1)) / (size*(size-1)));
    }

    public void calculateExponentialRankProbability() {
        this.probability = (1 - Math.exp(-1 * rank));
    }
    private void uniformMutation(Random rnd)
    {
        for (int i = 0; i < value.length; i++) {
            double r = rnd.nextDouble();
            if (r < mutationRate) {
                value[i] = rnd.nextDouble() * UB;
                if (rnd.nextBoolean()) {
                    value[i] *= -1;
                }
            }
        }
    }


    private void nonUniformMutation(Random rnd)
    {
        for (int i = 0; i < value.length; i++) {
            double h = rnd.nextGaussian() * sigma[0];
            value[i] = boundedAdd(value[i], h);
        }
    }

    private void uncorMutat(double epsilon, Random rnd)
    {
        double tau = 0.9;
        double gamma = tau * rnd.nextGaussian();
        sigma[0] *= Math.exp(gamma);
        sigma[0] = Math.max(sigma[0], epsilon);

        for (int i = 0; i < value.length; i++) {
            value[i] = boundedAdd(value[i], sigma[0] * rnd.nextGaussian());
        }
    }

    private void uncorMutWithNStepSizes(double epsilon, Random rnd,double t, double t2)
    {

        double gamma = t2 * rnd.nextGaussian();

        for (int i = 0; i < value.length; i++) {
            double g = rnd.nextGaussian();
            sigma[i] *= Math.exp(gamma + t * g);
            sigma[i] = Math.max(sigma[i], epsilon);
            value[i] = boundedAdd(value[i], sigma[i] * g);
        }
    }

    private void correlatedMutation(double epsilon, Random rnd)
    {
        double tau = 0.05;    // local learning rate
        double tau2 = 0.9;   // global learning rate

        double beta = 5;
        int n = value.length;
        int sign;
        int alpha_i;
        int n_alpha = n * (n - 1) / 2;

        double[] means = new double[n];
        double[] dx = new double[n];
        double gamma = tau2 * rnd.nextGaussian();

        // Java automatically initializes doubles with 0
        alpha = new double[n_alpha];
        cov = new double[n][n];

        for (int i = 0; i < n; i++) {
            double g = rnd.nextGaussian();
            sigma[i] *= Math.exp(gamma + tau * g);
            sigma[i] = Math.max(sigma[i], epsilon);

            for (int j = 0; j < n_alpha; j++) {
                alpha[j] += beta * rnd.nextGaussian();
                if (Math.abs(alpha[j]) > Math.PI) {
                    sign = (int) Math.signum(alpha[j]);
                    alpha[j] = alpha[j] - 2 * Math.PI * sign;
                }
            }
        }

        // construct covariance matrix
        alpha_i = 0;
        for (int x = 0; x < n; x++) {
            cov[x][x] = Math.pow(sigma[x], 2);
            for (int y = x + 1; y < n; y++) {
                cov[x][y] = 0.5 * (Math.pow(sigma[x], 2) - Math.pow(sigma[y], 2)) * Math.tan(2 * alpha[alpha_i]);
            }
            alpha_i++;
        }
        // values below the diagonal are the same as above
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < x; y++) {
                cov[x][y] = cov[y][x];
            }
        }

        dx = mvnd(means, cov,rnd);
       // dx = new MultivariateNormalDistribution(means, cov).sample();
        for (int i = 0; i < n; i++) {
            value[i] = boundedAdd(value[i], dx[i]);
        }
    }

    // check to ensure v stays in domain of function
    private double boundedAdd(double v, double dv)
    {
        if (dv < 0) {
            v = Math.max(v + dv, LB) ;
        } else if (dv > 0) {
            v = Math.min(v + dv, UB);
        }
        return v;
    }
/////////////////////////////////////////////////////////////

    private double[] mvnd(double[] menas,double[][] conv, Random rnd ){
        double[][] ch = cholesky(conv);
        double[][] z = generateStandardNormal( rnd);

        double[][] p =  multiplyByMatrix(ch,z);
        double[][] sum =  new double[3][10];
        //sum
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 10; j++) {
                sum[i][j] =  p[j][i] + menas[j];
            }
        }
        //should pick randomly
      return  sum[0];
    }

    public static double[][] multiplyByMatrix(double[][] m1, double[][] m2) {
        int m1ColLength = m1[0].length; // m1 columns length
        int m2RowLength = m2.length;    // m2 rows length
        if(m1ColLength != m2RowLength) return null; // matrix multiplication is not possible
        int mRRowLength = m1.length;    // m result rows length
        int mRColLength = m2[0].length; // m result columns length
        double[][] mResult = new double[mRRowLength][mRColLength];
        for(int i = 0; i < mRRowLength; i++) {         // rows from m1
            for(int j = 0; j < mRColLength; j++) {     // columns from m2
                for(int k = 0; k < m1ColLength; k++) { // columns from m1
                    mResult[i][j] += m1[i][k] * m2[k][j];
                }
            }
        }
        return mResult;
    }

    private double[][] generateStandardNormal( Random rnd){
        double[][] ret =  new double[10][3];
        for(int i=0; i <10;i++){
            for(int y=0; y<3;y++){
                ret[i][y]= rnd.nextGaussian();
            }
        }
        return ret;
    }
    private double[][] cholesky(double[][] A) {
        if (!isSquare(A)) {
            throw new RuntimeException("Matrix is not square");
        }
        if (!isSymmetric(A)) {
            throw new RuntimeException("Matrix is not symmetric");
        }

        int N  = A.length;
        double[][] L = new double[N][N];

        for (int i = 0; i < N; i++)  {
            for (int j = 0; j <= i; j++) {
                double sum = 0.0;
                for (int k = 0; k < j; k++) {
                    sum += L[i][k] * L[j][k];
                }
                if (i == j) L[i][i] = Math.sqrt(A[i][i] - sum);
                else        L[i][j] = 1.0 / L[j][j] * (A[i][j] - sum);
            }
            if (L[i][i] <= 0) {
                throw new RuntimeException("Matrix not positive definite");
            }
        }
        return L;
    }

    // is symmetric
    private boolean isSymmetric(double[][] A) {
        int N = A.length;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < i; j++) {
                if (A[i][j] != A[j][i]) return false;
            }
        }
        return true;
    }

    // is symmetric
    private boolean isSquare(double[][] A) {
        int N = A.length;
        for (int i = 0; i < N; i++) {
            if (A[i].length != N) return false;
        }
        return true;
    }

}
