
"""
Material properties of the glass LITHOSIL-Q from the Schott glass library.
Units are in Celsius, atmospheres, and meters.
"""
from numpy import sqrt


class BK7:
    t_ref = 20
    p_ref = 1

    # Sellmeier using the Sellmeier 1 dispersion formula in chapter 21 of the Zemax manual.
    k1 = 1.039612120
    l1 = 6.000698670e-3
    k2 = 2.317923440e-1
    l2 = 2.001791440e-2
    k3 = 1.010469450
    l3 = 1.035606530e2

    # Index thermal coefficients for change in index from the reference index.
    # These cefficients are used with equation [1] in chapter 22 of the Zemax user manual.
    d0 = 1.86000000e-6
    d1 = 1.31000000e-8
    d2 = -1.37000000e-11
    e0 = 4.34000000e-7
    e1 = 6.27000000e-10
    lam_tk = 1.7e-1

    # Coefficient of linear thermal expansion
    a = 7.1e-6

    def delta_n(self, t, n_ref, lam):
        """
        Calculate the change in refractive index, with respect to the reference index, as a function of temperature
        deviation from the reference temperature.

        This equation can be found in [1] as well as chapter 22 of the Zemax user manual.
        :param t: Float. The temperature at which to calculate the refractive index change.
        :param n_ref: Float. The reference refractive index at the reference temperature at the wavelength of interest.
        :param lam: Float. Wavelength of interest.
        :return: Float. The temperature derivative of the refractive index.
        """
        lam = lam * 1e6
        dt = t - self.t_ref
        return (n_ref ** 2 - 1) / (2 * n_ref) * (
                    self.d0 * dt + self.d1 * dt ** 2 + self.d2 * dt ** 3 + (self.e0 * dt + self.e1 * dt ** 2) / (lam ** 2 - self.lam_tk ** 2))

    def beta(self, t, n_ref, lam):
        """
        Calculate the temperature derivative of the refractive index of glass.

        :param t: Float. The temperature at which to calculate the refractive index change.
        :param n_ref: Float. The reference refractive index at the reference temperature at the wavelength of interest.
        :param lam: Float. Wavelength of interest.
        :return: Float. The temperature derivative of the refractive index.
        """
        lam = lam * 1e6
        dt = t - self.t_ref
        return (n_ref ** 2 - 1) / (2 * n_ref) * (
                    self.d0 + 2 * self.d1 * dt + 3 * self.d2 * dt ** 2 + (self.e0 + 2 * self.e1 * dt) / (lam ** 2 - self.lam_tk ** 2))

    def sellmeier_1(self, lam):
        """
        Calculate the index of refraction using the Sellmeier 1 dispersion formula in chapter 21 of the Zemax manual.

        These coefficients are measured at a particular reference temperature. To find the index at the wavelength of
        interest at a different temperture, use this formula in conjucntion with delta_n().
        :param lam: Float. The wavelength of interest.
        :return: Float. The index of refraction at the wavelength of interest.
        """
        lam = lam * 1e6
        return sqrt(
            1 + self.k1 * lam ** 2 / (lam ** 2 - self.l1) + self.k2 * lam ** 2 / (lam ** 2 - self.l2) + self.k3 * lam ** 2 / (lam ** 2 - self.l3))
