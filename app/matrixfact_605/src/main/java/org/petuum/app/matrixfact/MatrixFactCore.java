package org.petuum.app.matrixfact;

import org.petuum.app.matrixfact.Rating;
import org.petuum.app.matrixfact.LossRecorder;

import org.petuum.ps.PsTableGroup;
import org.petuum.ps.row.double_.DenseDoubleRow;
import org.petuum.ps.row.double_.DenseDoubleRowUpdate;
import org.petuum.ps.row.double_.DoubleRow;
import org.petuum.ps.row.double_.DoubleRowUpdate;
import org.petuum.ps.table.DoubleTable;
import org.petuum.ps.common.util.Timer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;

public class MatrixFactCore {
    private static final Logger logger =
        LoggerFactory.getLogger(MatrixFactCore.class);

    public static double dotProd(DoubleRow lCache, DoubleRow rCache, int K) {
        // Compute the dot produc to two rows
        double eij = 0.0;
        for (int col = 0; col < K; col++) {
            eij += (lCache.get(col) * rCache.get(col));
        }
        return eij;
    }

    // Perform a single SGD on a rating and update LTable and RTable
    // accordingly.
    public static void sgdOneRating(Rating r, double learningRate,
            DoubleTable LTable, DoubleTable RTable, int K, double lambda) {
        // TODO
        int lIdx   = r.userId;
        int rIdx   = r.prodId;
        int rating = r.rating;

        // Read in the left row
        DoubleRow lCache = new DenseDoubleRow(K + 1);
        DoubleRow lRow   = LTable.get(lIdx);
        lCache.reset(lRow);

        // Read in the right row
        DoubleRow rCache = new DenseDoubleRow(K + 1);
        DoubleRow rRow   = RTable.get(rIdx);
        rCache.reset(rRow);

        for (int col = 0; col < K + 1; col++) {
            lCache.getUnlocked(col);
            rCache.getUnlocked(col);
        }

        // Compute e_ij
        double eij = dotProd(lCache, rCache, K) - (double) rating;

        // get ni, mj
        int ni  = (int) lCache.get(K);
        int mj  = (int) rCache.get(K);

        assert (ni > 0) && (mj > 0);

        // Batch update
        DoubleRowUpdate lUpdates = new DenseDoubleRowUpdate(K + 1);
        DoubleRowUpdate rUpdates = new DenseDoubleRowUpdate(K + 1);
        for (int col = 0; col < K; col++) {
            double lGrad = 2 * learningRate *
              (eij * rCache.get(col) - lambda / (double) ni * lCache.get(col));

            double rGrad = 2 * learningRate *
              (eij * lCache.get(col) - lambda / (double) mj * rCache.get(col));

            lUpdates.setUpdate(col, lGrad);
            rUpdates.setUpdate(col, rGrad);
        }
        lUpdates.setUpdate(K, 0.0);
        rUpdates.setUpdate(K, 0.0);
        LTable.batchInc(lIdx, lUpdates);
        RTable.batchInc(rIdx, rUpdates);
    }

    // Evaluate square loss on entries [elemBegin, elemEnd), and L2-loss on of
    // row [LRowBegin, LRowEnd) of LTable,  [RRowBegin, RRowEnd) of Rtable.
    // Note the interval does not include LRowEnd and RRowEnd. Record the loss to
    // lossRecorder.
    public static void evaluateLoss(ArrayList<Rating> ratings, int ithEval,
            int elemBegin, int elemEnd, DoubleTable LTable,
            DoubleTable RTable, int LRowBegin, int LRowEnd, int RRowBegin,
            int RRowEnd, LossRecorder lossRecorder, int K, double lambda) {
        // TODO
        double sqLoss = 0;
        double totalLoss = 0;

        DoubleRow lCache = new DenseDoubleRow(K + 1);
        DoubleRow rCache = new DenseDoubleRow(K + 1);
        for (int i = elemBegin; i < elemEnd; i++) {
            Rating r = ratings.get(i);
            int lIdx   = r.userId;
            int rIdx   = r.prodId;
            int rating = r.rating;

            // Read in the left row
            DoubleRow lRow   = LTable.get(lIdx);
            lCache.reset(lRow);

            // Read in the right row
            DoubleRow rRow   = RTable.get(rIdx);
            rCache.reset(rRow);

            for (int col = 0; col < K; col++) {
                lCache.getUnlocked(col);
                rCache.getUnlocked(col);
            }

            // Compute e_ij
            double eij = dotProd(lCache, rCache, K) - (double) rating;
            sqLoss += (eij * eij);
        }

        for (int i = LRowBegin; i < LRowEnd; i++) {
            Double lRow = LTable.get(i);
            lCache.reset(lRow);
            for (int col = 0; col < K; col++)
                lCache.getUnlocked(col);

            for (int col = 0; col < K; col++) {
                double val = lCache.get(col);
                totalLoss += (val * val);
            }
        }

        for (int i = RRowBegin; i < RRowEnd; i++) {
            Double rRow = RTable.get(i);
            rCache.reset(rRow);
            for (int col = 0; col < K; col++)
                rCache.getUnlocked(col);

            for (int col = 0; col < K; col++) {
                double val = rCache.get(col);
                totalLoss += (val * val);
            }
        }

        totalLoss *= lambda;
        totalLoss += sqLoss;

        lossRecorder.incLoss(ithEval, "SquareLoss", sqLoss);
        lossRecorder.incLoss(ithEval, "FullLoss", totalLoss);
        lossRecorder.incLoss(ithEval, "NumSamples", elemEnd - elemBegin);
    }
}
