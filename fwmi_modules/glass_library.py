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


class P_SF68:
    t_ref = 20
    p_ref = 1

    # Sellmeier using the Sellmeier 1 dispersion formula in chapter 21 of the Zemax manual.
    k1 = 2.3330067
    l1 = 0.0168838419
    k2 = 0.452961396
    l2 = 0.0716086325
    k3 = 1.25172339
    l3 = 118.707479

    # Index thermal coefficients for change in index from the reference index.
    # These cefficients are used with equation [1] in chapter 22 of the Zemax user manual.
    d0 = 1.55e-5
    d1 = 2.3e-8
    d2 = -3.46e-11
    e0 = 2.76e-6
    e1 = 2.93e-9
    lam_tk = 297e-3

    # Coefficient of linear thermal expansion
    a = 8.43e-6

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
                self.d0 * dt + self.d1 * dt ** 2 + self.d2 * dt ** 3 + (self.e0 * dt + self.e1 * dt ** 2) / (
                    lam ** 2 - self.lam_tk ** 2))

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
                self.d0 + 2 * self.d1 * dt + 3 * self.d2 * dt ** 2 + (self.e0 + 2 * self.e1 * dt) / (
                    lam ** 2 - self.lam_tk ** 2))

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
            1 + self.k1 * lam ** 2 / (lam ** 2 - self.l1) + self.k2 * lam ** 2 / (
                        lam ** 2 - self.l2) + self.k3 * lam ** 2 / (lam ** 2 - self.l3))


class LITHOSIL_Q:
    t_ref = 20
    p_ref = 1

    # Sellmeier using the Sellmeier 1 dispersion formula in chapter 21 of the Zemax manual.
    k1 = 0.67071081
    l1 = 4.49192312e-3
    k2 = 4.33322857e-1
    l2 = 1.32812976e-2
    k3 = 8.77379057e-1
    l3 = 9.58899878e1

    # Index thermal coefficients for change in index from the reference index.
    # These cefficients are used with equation [1] in chapter 22 of the Zemax user manual.
    d0 = 2.06e-5
    d1 = 2.51e-8
    d2 = -2.47e-11
    e0 = 3.12e-7
    e1 = 4.22e-10
    lam_tk = 1.6e-1

    # Coefficient of linear thermal expansion
    a = 0.5e-6

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
