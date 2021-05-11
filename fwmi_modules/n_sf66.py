
"""
Material properties of the glass N-SF66 from the Schott glass library.
Units are in Celsius, atmospheres, and meters.
"""
from numpy import sqrt


class N_SF66:
    t_ref = 20
    p_ref = 1

    # Sellmeier using the Sellmeier 1 dispersion formula in chapter 21 of the Zemax manual.
    k1 = 2.024597
    l1 = 0.0147053225
    k2 = 0.470187196
    l2 = 0.0692998276
    k3 = 2.59970433
    l3 = 161.817601

    # Index thermal coefficients for change in index from the reference index.
    # These cefficients are used with equation [1] in chapter 22 of the Zemax user manual.
    d0 = -4.3e-6
    d1 = 1.15e-8
    d2 = 4.31e-11
    e0 = 9.62e-7
    e1 = 1.62e-9
    lam_tk = 3.22e-1

    # Coefficient of linear thermal expansion
    a = 5.9e-6

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
