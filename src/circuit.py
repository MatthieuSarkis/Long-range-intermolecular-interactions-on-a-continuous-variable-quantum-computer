from strawberryfields import ops

class Circuit():

    def __init__(self, qnn, sf_params, layers):

        self.qnn = qnn
        self.sf_params = sf_params

        with self.qnn.context as q:
            for k in range(layers):
                self.layer(sf_params[k], q)

    def interferometer(self, params, q):
        """Parameterised interferometer acting on ``N`` modes.

        Args:
            params (list[float]): list of length ``max(1, N-1) + (N-1)*N`` parameters.

                * The first ``N(N-1)/2`` parameters correspond to the beamsplitter angles
                * The second ``N(N-1)/2`` parameters correspond to the beamsplitter phases
                * The final ``N-1`` parameters correspond to local rotation on the first N-1 modes

            q (list[RegRef]): list of Strawberry Fields quantum registers the interferometer
                is to be applied to
        """
        N = len(q)
        theta = params[:N*(N-1)//2]
        phi = params[N*(N-1)//2:N*(N-1)]
        rphi = params[-N+1:]

        if N == 1:
            # the interferometer is a single rotation
            ops.Rgate(rphi[0]) | q[0]
            return

        n = 0  # keep track of free parameters

        # Apply the rectangular beamsplitter array
        # The array depth is N
        for l in range(N):
            for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
                # skip even or odd pairs depending on layer
                if (l + k) % 2 != 1:
                    ops.BSgate(theta[n], phi[n]) | (q1, q2)
                    n += 1

        # apply the final local phase shifts to all modes except the last one
        for i in range(max(1, N - 1)):
            ops.Rgate(rphi[i]) | q[i]

    def layer(self, params, q):
        """CV quantum neural network layer acting on ``N`` modes.

        Args:
            params (list[float]): list of length ``2*(max(1, N-1) + N**2 + n)`` containing
                the number of parameters for the layer
            q (list[RegRef]): list of Strawberry Fields quantum registers the layer
                is to be applied to
        """
        N = len(q)
        M = int(N * (N - 1)) + max(1, N - 1)

        int1 = params[:M]
        s = params[M:M+N]
        int2 = params[M+N:2*M+N]
        dr = params[2*M+N:2*M+2*N]
        dp = params[2*M+2*N:2*M+3*N]
        k = params[2*M+3*N:2*M+4*N]

        # begin layer
        self.interferometer(int1, q)

        for i in range(N):
            ops.Sgate(s[i]) | q[i]

        self.interferometer(int2, q)

        for i in range(N):
            ops.Dgate(dr[i], dp[i]) | q[i]
            ops.Kgate(k[i]) | q[i]