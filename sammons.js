function sammon(x, n = 2, display = 0, inputdist = 'raw', maxhalves = 20, maxiter = 500, tolfun = 1e-9, init = 'pca') {

    const X = x;

    // import numpy as np
    // from scipy.spatial.distance import cdist
    // In JavaScript, we can use math.js library for vector/matrix calculations
    const math = require('mathjs');

    // Create distance matrix unless given by parameters
    let xD;
    if (inputdist === 'distance') {
        xD = X;
    } else {
        xD = math.distance(X, X, 2); // 2 denotes Euclidean distance
    }

    // Remaining initialisation
    const N = X.length;
    const scale = 0.5 / math.sum(xD);

    let Y;
    if (init === 'pca') {
        const [UU, DD, _] = math.lusolve(X, math.eye(N));
        Y = math.multiply(UU, math.subset(DD, math.index(math.range(0, n))));
    } else {
        Y = math.random([N, n]);
    }
    const one = math.ones([N, n]);

    xD = math.add(xD, math.eye(N));
    let xDinv = math.dotPow(xD, -1); // Returns Infinity where D = 0.
    xDinv = math.subset(xDinv, math.index(math.isNaN(xDinv)), 0); // Fix by replacing NaN with 0.
    xDinv = math.subset(xDinv, math.index(math.isInfinity(xDinv)), 0); // Fix by replacing Infinity with 0.

    let yD = math.distance(Y, Y, 2);
    yD = math.add(yD, math.eye(N));
    let yDinv = math.dotPow(yD, -1); // Returns Infinity where d = 0. 
    yDinv = math.subset(yDinv, math.index(math.isNaN(yDinv)), 0); // Fix by replacing NaN with 0.
    yDinv = math.subset(yDinv, math.index(math.isInfinity(yDinv)), 0); // Fix by replacing Infinity with 0.

    xD = math.subset(xD, math.index(math.range(0, N), math.range(0, N)), 1);
    yD = math.subset(yD, math.index(math.range(0, N), math.range(0, N)), 1);
    xDinv = math.subset(xDinv, math.index(math.range(0, N), math.range(0, N)), 0);
    yDinv = math.subset(yDinv, math.index(math.range(0, N), math.range(0, N)), 0);

    const delta = math.subtract(xD, yD);
    let E = math.dotPow(delta, 2);
    E = math.multiply(E, xDinv);
    E = math.sum(E);

    // Get on with it
    for (let i = 0; i < maxiter; i++) {

        // Compute gradient, Hessian and search direction (note it is actually
        // 1/4 of the gradient and Hessian, but the step size is just the ratio
        // of the gradient and the diagonal of the Hessian so it doesn't
        // matter).
        const delta = math.subtract(yDinv, xDinv);
        const deltaone =
