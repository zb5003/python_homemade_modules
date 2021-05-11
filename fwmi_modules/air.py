class Air:
    # Coefficient of linear thermal expansion
    a = 0

    def n_air(self, lam, t, p):
        """
        Calculate the index of refraction of air.

        :param lam: Float. The wavelength at which to calculate the refractive index, in meters.
        :param t: Float. The temperature, in celsius, at which to calculate the refractive index.
        :param p: Float. The pressure, in atmospheres, at which to calculate the refractive index.
        :return: Float. The refractive index of air.
        """
        lam = lam * 1e6
        return 1 + (6432.8 + 2949810 * lam**2 / (146 * lam**2 - 1) + 25540 * lam**2 / (41 * lam**2 - 1)) / (1 + (t - 15) * 3.4785e-3) * 1e-8 * p

    def beta_air(self, lam, t, p):
        """
        Calculate the temperature derivative of the refractive index of air.

        :param lam: Float. The wavelength at which to calculate the refractive index, in meters.
        :param t: Float. The temperature, in celsius, at which to calculate the refractive index.
        :param p: Float. The pressure, in atmospheres, at which to calculate the refractive index.
        :return: Float. dn_air / dT
        """
        lam = lam * 1e6
        return -3.4785e-3 / (1 + (t - 15) * 3.4785e-3) * (self.n_air(lam, t, p) - 1)