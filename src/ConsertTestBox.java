//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//

import org.vu.contest.ContestEvaluation;
import org.vu.contest.ContestSubmission;

import java.util.Date;

//import org.vu.contest.ContestEvaluation;

public class ConsertTestBox {
    public ConsertTestBox() {
    }

    public static void main(String[] var0) {
        String var1 = null;
        String var2 = null;
        long var3 = -1L;
        long var5 = -1L;
        boolean var7 = false;
        String[] var8 = var0;
        int var9 = var0.length;

        for(int var10 = 0; var10 < var9; ++var10) {
            String var11 = var8[var10];
            if (var11.startsWith("-submission=")) {
                var1 = var11.split("=")[1];
            } else if (var11.startsWith("-evaluation=")) {
                var2 = var11.split("=")[1];
            } else if (var11.startsWith("-seed=")) {
                var5 = Long.parseLong(var11.split("=")[1]);
            } else if (var11.equals("-nosec")) {
                var7 = true;
            } else {
                System.out.println("Invalid flag: '" + var11 + "' !");
            }
        }

        if (var1 == null) {
            throw new Error("Submission ID was not specified! Cannot run...\n Use -submission=<classnamehere> to specify the name of the algorithm class.");
        } else if (var2 == null) {
            throw new Error("Evaluation ID was not specified! Cannot run...\n Use -evaluation=<classnamehere> to specify the name of the evaluation class.");
        } else if (var5 < 0L) {
            throw new Error("Seed was not specified! Cannot run...\n Use -seed=<number> to specify the random seed.");
        } else {
            Class var23 = null;

            try {
                var23 = Class.forName(var2);
            } catch (Throwable var22) {
                System.err.println("Could not load evaluation class for evaluation '" + var2 + "'");
                var22.printStackTrace();
                System.exit(1);
            }

            ContestEvaluation var24 = null;

            try {
                var24 = (ContestEvaluation)var23.newInstance();
            } catch (Throwable var21) {
                System.err.println("ExecutionError: Could not instantiate evaluation object for evaluation '" + var2 + "'");
                var21.printStackTrace();
                System.exit(1);
            }

            Class var25 = null;

            try {
                var25 = Class.forName(var1);
            } catch (Throwable var20) {
                System.err.println("ExecutionError: Could not load submission class for player '" + var1 + "'");
                var20.printStackTrace();
                System.exit(1);
            }

            if (!var7) {
                ConsertTestSecurity var26 = new ConsertTestSecurity((String)null);
                var26.addPermittedRead(System.getProperty("user.dir") + "/submission.jar");
                var26.addPermittedRead(System.getProperty("java.home") + "/lib");
                var26.addPermittedRead(System.getProperty("java.home") + "/lib/resources.jar");
                var26.addPermittedRead(System.getProperty("java.home") + "/lib/meta-index");
                var26.addPermittedRead(System.getProperty("java.home") + "/lib/rt.jar");
                var26.addPermittedRead(System.getProperty("java.home") + "/lib/sunrsasign.jar");
                var26.addPermittedRead(System.getProperty("java.home") + "/lib/jsse.jar");
                var26.addPermittedRead(System.getProperty("java.home") + "/lib/jce.jar");
                var26.addPermittedRead(System.getProperty("java.home") + "/lib/charsets.jar");
                var26.addPermittedRead(System.getProperty("java.home") + "/lib/jfr.jar");
                var26.addPermittedRead(System.getProperty("java.home") + "/lib/jfxrt.jar");
                var26.addPermittedRead(System.getProperty("java.home") + "/lib/alt-rt.jar");
                var26.addPermittedRead(System.getProperty("java.home") + "/classes");
                var26.addPermittedRead(System.getProperty("java.home") + "/meta-index");

                //our classes
                var26.addPermittedRead(System.getProperty("user.dir")+"\\Options.class");
                var26.addPermittedRead(System.getProperty("user.dir")+"\\Options$Mutation.class");
                var26.addPermittedRead(System.getProperty("user.dir")+"\\Options$ParentSelection.class");
                var26.addPermittedRead(System.getProperty("user.dir")+"\\Options$Recombination.class");
                var26.addPermittedRead(System.getProperty("user.dir")+"\\Options$SurvivorSelection.class");
                var26.addPermittedRead(System.getProperty("user.dir")+"\\Individual$1.class");
                var26.addPermittedRead(System.getProperty("user.dir")+"\\Individual.class");
                var26.addPermittedRead(System.getProperty("user.dir")+"\\IslandModel.class");
                var26.addPermittedRead(System.getProperty("user.dir")+"\\Population$1.class");
                var26.addPermittedRead(System.getProperty("user.dir")+"\\Population.class");

                var26.addPermittedRead(System.getProperty("java.home") + "/ext/localedata.jar");
                //Options$Mutation

                System.setSecurityManager(var26);
            }

            ContestSubmission var27 = null;

            try {
                var27 = (ContestSubmission)var25.newInstance();
            } catch (Throwable var19) {
                System.err.println("ExecutionError: Could not instantiate submission object for player '" + var1 + "'");
                var19.printStackTrace();
                System.exit(1);
            }

            var27.setSeed(var5);
            var27.setEvaluation(var24);
            Date var12 = new Date();
            long var13 = var12.getTime();

            try {
                var27.run();
            } catch (SecurityException var17) {
                var17.printStackTrace();
                System.out.println("Your code has attempted a security violation!");
                System.out.println("This would terminate execution (thus no score would be assigned!)");
                System.exit(0);
            } catch (Throwable var18) {
                var18.printStackTrace();
                System.out.println("Your code has thrown an Exception/Error!");
                System.out.println("This would halt execution (the best score achieved so far would be assigned)");
            }

            var12 = new Date();
            long var15 = var12.getTime() - var13;
            System.out.println("Score: " + Double.toString(var24.getFinalResult()));
            System.out.println("Runtime: " + var15 + "ms");
            System.exit(0);
        }
    }
}
