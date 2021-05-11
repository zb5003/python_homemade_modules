"""
Classes necessary to design a FWMI.
Equations come from
    [1] Z. Cheng, et al. Opt. Exp. 23, 12117 (2015).
    [2] Zemax User Manual.

All angles are in radians.
"""
import numpy as np
from scipy.integrate import nquad
import numpy.linalg as la
from scipy import constants as const
from glass_library import *
from air import Air


class Hybrid:
    """
    Parent class for the hybrid field-widened Michelson interferometer (FWMI).

    One arm is pure glass, the other is part glass and part air. The glass in the pure arm
    is denoted by the index 1, the glass in the hybid arm is denoted by 2, and the air in
    the hybrid arm is denoted by 3.
    """

    def __init__(self, fopd, theta_t, gamma_m, gamma_a, lam, t_ref, p, pure_arm=N_SF66, hybrid_arm=P_SF68):
        """

        :param fopd:
        :param theta_t:
        :param gamma_m:
        :param gamma_a:
        :param lam:
        :param t_ref:
        :param p:
        """
        self.fopd = fopd
        self.theta_t = theta_t
        self.gamma_m = gamma_m
        self.gamma_a = gamma_a
        self.lam = lam
        self.t_ref = t_ref
        self.p = p

        self.nu = const.c / self.lam

    def fsr(self, opd):
        """
        Calculate the free spectral range of the MI.

        :param opd: Float. The optical path difference.
        :return: Float. The free spectral range.
        """
        return const.c / opd

    def snell_cos_to_sin(self, n, theta):
        """
        Go from cos to sin using Snell's law.

        For the FWMI, one of the materials is always going to be air.
        :param n: Float. The index of refraction.
        :param theta: Float. The angle of incidence on the FWMI.
        :return: Float. Cos in terms of n and sin.
        """
        return np.sqrt(1 - np.sin(theta)**2 / n**2)

    def transmittance_simple(self, gamma, fopd):
        """
        Calculate the FWMI transmittance with no tilt.

        Eq. 24 in [1].
        :param gamma: Float. The spectral width of the incident signal.
        :param fopd: Float. The fixed optical path difference.
        :return: Float. The transmittance of the FWMI.
        """
        return 1 / 2 - 1 / 2 * np.exp(-(np.pi * gamma / (const.c / fopd)) ** 2)

    def thermal_expansion(self, t, t_ref, alpha):
        """
        Calculate the linear expansion of a material, (1 + alpha * dt), based on L = L0(1 + alpha * (t - t_ref))).

        :param t: Float. The temperature of the material.
        :param t_ref: Float. The reference temperature (often 20 Celsius).
        :param alpha: Float. The linear thermal expansion coefficient of the material.
        :return: Float. The expansion of the material.
        """
        pass

    def opd_exact(self, theta, n1, d1, n2, d2, n3, d3):
        """
        Calculate the exact optical path difference for the hybrid FWMI.

        Eq. 2 in [1].

        When theta = theta_t, the tilt angle, then this function gives the fixed optical path difference (FOPD).
        :param theta: Float. The incident angle on the FWMI.
        :param n1: Float. The index of refraction for the pure glass arm.
        :param d1: Float. The thickness of the pure glass arm.
        :param n2: Float. The index of refraction for the glass part of the hybrid arm.
        :param d2: Float. The thickness of the glass part of the hybrid arm.
        :param n3: Float. The index of refraction for the air part of the hybrid arm.
        :param d3: Float. The thickness of the air part of the hybrid arm.
        :return: Float. The exact optical path difference.
        """
        return 2 * (n1 * d1 * self.snell_cos_to_sin(n1, theta)
                    - n2 * d2 * self.snell_cos_to_sin(n2, theta)
                    - n3 * d3 * self.snell_cos_to_sin(n3, theta))

    def sdr_simple(self):
        """
        Calculate the spectral discrimination ratio with no tilt.

        A version of Eq. 17 in [1].
        :param gamma_m: Float. The spectral width of the molecular signal.
        :param gamma_a: Float. The spectral width of the aerosol signal.
        :param fopd: Float. The fixed optical path difference.
        :return: Float. The spectral discrimination ratio.
        """
        return self.transmittance_simple(self.gamma_m, self.fopd) / self.transmittance_simple(self.gamma_a, self.fopd)

    def delta_phi(self, opd_theta, opd_theta_t):
        """
        Calculate the phase difference between a ray incident at angle theta and a ray at angle theta_t.

        :param opd_theta: Float. The optical path difference for the non-central incident angle.
        :param opd_theta_t: Float. The optical path difference for the central (tilt) incident angle.
        :return: Float. The phase difference.
        """
        return 2 * np.pi * self.nu * (opd_theta - opd_theta_t) / const.c


class Pure(Hybrid):

    def __init__(self, fopd, theta_t, gamma_m, gamma_a, lam,t_ref, p):
        """

        :param fopd:
        :param theta_t:
        :param gamma_m:
        :param gamma_a:
        :param lam:
        :param t:
        :param t_ref:
        :param p:
        """
        Hybrid.__init__(self, fopd, theta_t, gamma_m, gamma_a, lam, t_ref, p)

    def opd_exact_pure(self, theta, n1, d1, n2, d2):
        """
        Calculate the exact OPD for a pure FWMI.

        :param theta:
        :param n1:
        :param d1:
        :param n2:
        :param d2:
        :return:
        """
        return self.opd_exact(theta, n1, d1, n2, d2, n3=1, d3=0)


class H1(Hybrid):

    def __init__(self, fopd, theta_t, gamma_m, gamma_a, lam, t, t_ref, p, d_opd_d_t=0):
        """

        :param fopd:
        :param theta_t:
        :param gamma_m:
        :param gamma_a:
        :param lam:
        :param t:
        :param t_ref:
        :param p:
        """
        Hybrid.__init__(self, fopd, theta_t, gamma_m, gamma_a, lam, t_ref, p)
        self.t = t
        self.d_opd_d_t = d_opd_d_t
        self.pure = N_SF66()
        self.n_ref_1 = self.pure.sellmeier_1(self.lam)
        self.n1 = self.generate_n1(self.t)
        self.b1 = self.pure.beta(self.t, self.n_ref_1, self.lam)

        self.hybrid = P_SF68()
        self.n_ref_2 = self.hybrid.sellmeier_1(self.lam)
        self.n2 = self.generate_n2(self.t)
        self.b2 = self.hybrid.beta(self.t, self.n_ref_2, self.lam)

        self.air = Air()
        self.n3 = self.air.n_air(self.lam, self.t, self.p)
        self.b3 = self.air.beta_air(self.lam, self.t, self.p)
        # self.thickness_generator = np.asarray([self.fopd, 0, 1064e-9/10])  #[FOPD, w(theta), d_OPD/d_T]

        self.d1, self.d2, self.d3 = self.generate_thickness()

    def generate_n1(self, t):
        """

        :param t:
        :return:
        """
        return self.n_ref_1 + self.pure.delta_n(t, self.n_ref_1, self.lam)

    def generate_n2(self, t):
        """

        :param t:
        :return:
        """
        return self.n_ref_2 + self.hybrid.delta_n(t, self.n_ref_2, self.lam)

    def generate_n3(self, t):
        """

        :param t:
        :return:
        """
        return self.air.n_air(self.lam, t, self.p)

    def generate_thickness(self):
        """
        Calculate the thicknesses of the three materials.

        Uses Eq 2, 6, and 7 from [1].
        :return: Tuple. The thicknesses (d1, d2, d3).
        """
        root1 = np.sqrt(self.n1 ** 2 - np.sin(self.theta_t) ** 2)
        root2 = np.sqrt(self.n2 ** 2 - np.sin(self.theta_t) ** 2)
        root3 = np.sqrt(self.n3 ** 2 - np.sin(self.theta_t) ** 2)

        m1 = 2 * np.asarray([[self.n1 * np.sqrt(1 - (np.sin(self.theta_t) / self.n1) ** 2), -self.n2 * np.sqrt(1 - (np.sin(self.theta_t) / self.n2) ** 2),
                           -self.n3 * np.sqrt(1 - (np.sin(self.theta_t) / self.n3) ** 2)],
                          [-1 / (2 * root1), 1 / (2 * root2), 1 / (2 * root3)],
                          [self.pure.a * root1 + self.n1 * self.b1 / root1, -(self.hybrid.a * root2 + self.n2 * self.b2 / root2),
                           -(self.air.a * root3 + self.n3 * self.b3 / root3)]])
        v1 = np.asarray([self.fopd, 0, self.d_opd_d_t])

        return tuple(la.solve(m1, v1))

    def d1_thermal_expansion(self, t):
        """
        Calculate the thermal expansion of the pure glass arm.

        Because d1 was determined with self.t, the reference temperature used here is self.t, not self.pure.t_ref.
        :param t: Float. The temperature at which to calculate the thermal expansion.
        :return: Float. The thermally expanded length of the glass.
        """
        return self.d1 * (1 + self.pure.a * (t - self.t))

    def d2_thermal_expansion(self, t):
        """
        Calculate the thermal expansion of the glass in the hybrid arm.

        Because d2 was determined with self.t, the reference temperature used here is self.t, not self.hybrid.t_ref.
        :param t: Float. The temperature at which to calculate the thermal expansion.
        :return: Float. The thermally expanded length of the glass.
        """
        return self.d2 * (1 + self.hybrid.a * (t - self.t))

    def local_transmittance(self, d_phi, gamma, fsr, rms=0, phase_dev=0):
        """
        Calculate the local transmittance.

        For a given FWMI and incident beam, the output is a function of the angle in the plane of the FWMI.
        The incident intensities are assumed to both be 1.
        Eq. 14 in [1].

        The rms wavefront error can be incorporated for small errors by setting the rms argument.
        The wavefront error has a general form for different sources of error, see section 3.2.2c and Eq. 23 of [1].
        :param d_phi: Float. The phase difference between a ray incident at some angle and a ray incident at the tilt angle.
        :param gamma: Float. The linewidth of the signal (molecular or aerosol).
        :param fsr: Float. The FSR of the FWMI. A function of the tilt angle.
        :param rms: Float. The rms wavefront error. Default is 0.
        :param phase_dev: Float. The frequency shift from a perfect lock.
        :return: Float. The local transmittance.
        """
        rms = 2 * np.pi * self.nu * rms / const.c
        return 1 / 2 - 1 / 2 * np.exp(-(np.pi * gamma / fsr)**2) * np.cos(d_phi + phase_dev * 2 * np.pi / fsr) * (1 - rms**2 / 2)

    def mapping_angle(self, rho, phi, f):
        """
        Calculate the angle used in the transmittance map function.

        The map is from the exit angle of the FWMI to the focal plane of the exit lens, Eq. 16 in [1].
        :param rho: Float. The radial coordinate in the exit lens's image plane.
        :param phi: Float. The azimuthal coordinate in the exit lens's image plane.
        :param theta_t: Float. The tilt angle.
        :param f: Float. The exit lens's focal length.
        :return: Float. The angle to use with the local transmittance mapping function.
        """
        return np.arccos((2 * f * np.cos(self.theta_t)**2 - rho * np.sin(2 * self.theta_t) * np.cos(phi)) / (2 * np.sqrt(f**2 + rho**2) * np.cos(self.theta_t)))

    def transmittance_map_integrand(self, rho, phi, f, gamma, fsr, rms=0, phase_dev=False):
        """
        Calculate the integrand of the overall transmittance function.

        Based on the map from the local transmittance to the transmittance at the exit lens image plane.
        :param rho: Float. The radial coordinate in the exit lens's image plane.
        :param phi: Float. The azimuthal coordinate in the exit lens's image plane.
        :param f: Float. The exit lens's focal length.
        :param gamma: Float. The linewidth of the signal (molecular or aerosol).
        :param fsr: Float. The FSR of the FWMI. A function of the tilt angle.
        :param rms: Float. The rms wavefront error. Default is 0.
        :param phase_dev: Float. The frequency shift from a perfect lock.
        :return: Float. The integrand of the overall transmittance.
        """
        theta = self.mapping_angle(rho, phi, f)
        opd_theta = self.opd_exact(theta, self.n1, self.d1, self.n2, self.d2, self.n3, self.d3)
        del_phi = self.delta_phi(opd_theta, const.c / fsr)  # const.c / fsr instead of fopd to make changing fopd easier
        return rho * self.local_transmittance(del_phi, gamma, fsr, rms, phase_dev)

    def overall_transmittance(self, theta_d, f, gamma, fsr, rms=0, phase_dev=False):
        """
        Calculate the overall transmittance in the focal plane of the exit lens.

        Eq. 15 in [1].
        :param theta_d: Float. The HALF divergent angle that sets the limit on the integrand.
        :param f: Float. The exit lens's focal length.
        :param gamma: Float. The linewidth of the signal (molecular or aerosol).
        :param fsr: Float. The FSR of the FWMI. A function of the tilt angle.
        :param rms: Float. The rms wavefront error. Default is 0.
        :param phase_dev: Float. The frequency shift from a perfect lock.
        :return:
        """
        integral = nquad(self.transmittance_map_integrand, [[0, f * theta_d], [-np.pi, np.pi]], [f, gamma, fsr, rms, phase_dev])[0]
        return integral / (np.pi * f**2 * theta_d**2)


class all_silica(Hybrid):

    def __init__(self, fopd, theta_t, gamma_m, gamma_a, lam, t, t_ref, p, d_opd_d_t=0):
        """

        :param fopd:
        :param theta_t:
        :param gamma_m:
        :param gamma_a:
        :param lam:
        :param t:
        :param t_ref:
        :param p:
        """
        Hybrid.__init__(self, fopd, theta_t, gamma_m, gamma_a, lam, t_ref, p)
        self.t = t
        self.d_opd_d_t = d_opd_d_t
        self.pure = LITHOSIL_Q()
        self.n_ref_1 = self.pure.sellmeier_1(self.lam)
        self.n1 = self.generate_n1(self.t)
        self.b1 = self.pure.beta(self.t, self.n_ref_1, self.lam)

        self.hybrid = LITHOSIL_Q()
        self.n_ref_2 = self.hybrid.sellmeier_1(self.lam)
        self.n2 = self.generate_n2(self.t)
        self.b2 = self.hybrid.beta(self.t, self.n_ref_2, self.lam)

        self.air = Air()
        self.n3 = self.air.n_air(self.lam, self.t, self.p)
        self.b3 = self.air.beta_air(self.lam, self.t, self.p)
        # self.thickness_generator = np.asarray([self.fopd, 0, 1064e-9/10])  #[FOPD, w(theta), d_OPD/d_T]

        self.d1, self.d2, self.d3 = self.generate_thickness()

    def generate_n1(self, t):
        """

        :param t:
        :return:
        """
        return self.n_ref_1 + self.pure.delta_n(t, self.n_ref_1, self.lam)

    def generate_n2(self, t):
        """

        :param t:
        :return:
        """
        return self.n_ref_2 + self.hybrid.delta_n(t, self.n_ref_2, self.lam)

    def generate_n3(self, t):
        """

        :param t:
        :return:
        """
        return self.air.n_air(self.lam, t, self.p)

    def generate_thickness(self):
        """
        Calculate the thicknesses of the three materials.

        Uses Eq 2, 6, and 7 from [1].
        :return: Tuple. The thicknesses (d1, d2, d3).
        """
        root1 = np.sqrt(self.n1 ** 2 - np.sin(self.theta_t) ** 2)
        root2 = np.sqrt(self.n2 ** 2 - np.sin(self.theta_t) ** 2)
        root3 = np.sqrt(self.n3 ** 2 - np.sin(self.theta_t) ** 2)

        m1 = 2 * np.asarray([[self.n1 * np.sqrt(1 - (np.sin(self.theta_t) / self.n1) ** 2), -self.n2 * np.sqrt(1 - (np.sin(self.theta_t) / self.n2) ** 2),
                           -self.n3 * np.sqrt(1 - (np.sin(self.theta_t) / self.n3) ** 2)],
                          [-1 / (2 * root1), 1 / (2 * root2), 1 / (2 * root3)],
                          [self.pure.a * root1 + self.n1 * self.b1 / root1, -(self.hybrid.a * root2 + self.n2 * self.b2 / root2),
                           -(self.air.a * root3 + self.n3 * self.b3 / root3)]])
        v1 = np.asarray([self.fopd, 0, self.d_opd_d_t])

        return tuple(la.solve(m1, v1))

    def d1_thermal_expansion(self, t):
        """
        Calculate the thermal expansion of the pure glass arm.

        Because d1 was determined with self.t, the reference temperature used here is self.t, not self.pure.t_ref.
        :param t: Float. The temperature at which to calculate the thermal expansion.
        :return: Float. The thermally expanded length of the glass.
        """
        return self.d1 * (1 + self.pure.a * (t - self.t))

    def d2_thermal_expansion(self, t):
        """
        Calculate the thermal expansion of the glass in the hybrid arm.

        Because d2 was determined with self.t, the reference temperature used here is self.t, not self.hybrid.t_ref.
        :param t: Float. The temperature at which to calculate the thermal expansion.
        :return: Float. The thermally expanded length of the glass.
        """
        return self.d2 * (1 + self.hybrid.a * (t - self.t))

    def local_transmittance(self, d_phi, gamma, fsr, rms=0, phase_dev=0):
        """
        Calculate the local transmittance.

        For a given FWMI and incident beam, the output is a function of the angle in the plane of the FWMI.
        The incident intensities are assumed to both be 1.
        Eq. 14 in [1].

        The rms wavefront error can be incorporated for small errors by setting the rms argument.
        The wavefront error has a general form for different sources of error, see section 3.2.2c and Eq. 23 of [1].
        :param d_phi: Float. The phase difference between a ray incident at some angle and a ray incident at the tilt angle.
        :param gamma: Float. The linewidth of the signal (molecular or aerosol).
        :param fsr: Float. The FSR of the FWMI. A function of the tilt angle.
        :param rms: Float. The rms wavefront error. Default is 0.
        :param phase_dev: Float. The frequency shift from a perfect lock.
        :return: Float. The local transmittance.
        """
        rms = 2 * np.pi * self.nu * rms / const.c
        return 1 / 2 - 1 / 2 * np.exp(-(np.pi * gamma / fsr)**2) * np.cos(d_phi + phase_dev * 2 * np.pi / fsr) * (1 - rms**2 / 2)

    def mapping_angle(self, rho, phi, f):
        """
        Calculate the angle used in the transmittance map function.

        The map is from the exit angle of the FWMI to the focal plane of the exit lens, Eq. 16 in [1].
        :param rho: Float. The radial coordinate in the exit lens's image plane.
        :param phi: Float. The azimuthal coordinate in the exit lens's image plane.
        :param theta_t: Float. The tilt angle.
        :param f: Float. The exit lens's focal length.
        :return: Float. The angle to use with the local transmittance mapping function.
        """
        return np.arccos((2 * f * np.cos(self.theta_t)**2 - rho * np.sin(2 * self.theta_t) * np.cos(phi)) / (2 * np.sqrt(f**2 + rho**2) * np.cos(self.theta_t)))

    def transmittance_map_integrand(self, rho, phi, f, gamma, fsr, rms=0, phase_dev=False):
        """
        Calculate the integrand of the overall transmittance function.

        Based on the map from the local transmittance to the transmittance at the exit lens image plane.
        :param rho: Float. The radial coordinate in the exit lens's image plane.
        :param phi: Float. The azimuthal coordinate in the exit lens's image plane.
        :param f: Float. The exit lens's focal length.
        :param gamma: Float. The linewidth of the signal (molecular or aerosol).
        :param fsr: Float. The FSR of the FWMI. A function of the tilt angle.
        :param rms: Float. The rms wavefront error. Default is 0.
        :param phase_dev: Float. The frequency shift from a perfect lock.
        :return: Float. The integrand of the overall transmittance.
        """
        theta = self.mapping_angle(rho, phi, f)
        opd_theta = self.opd_exact(theta, self.n1, self.d1, self.n2, self.d2, self.n3, self.d3)
        del_phi = self.delta_phi(opd_theta, const.c / fsr)  # const.c / fsr instead of fopd to make changing fopd easier
        return rho * self.local_transmittance(del_phi, gamma, fsr, rms, phase_dev)

    def overall_transmittance(self, theta_d, f, gamma, fsr, rms=0, phase_dev=False):
        """
        Calculate the overall transmittance in the focal plane of the exit lens.

        Eq. 15 in [1].
        :param theta_d: Float. The HALF divergent angle that sets the limit on the integrand.
        :param f: Float. The exit lens's focal length.
        :param gamma: Float. The linewidth of the signal (molecular or aerosol).
        :param fsr: Float. The FSR of the FWMI. A function of the tilt angle.
        :param rms: Float. The rms wavefront error. Default is 0.
        :param phase_dev: Float. The frequency shift from a perfect lock.
        :return:
        """
        integral = nquad(self.transmittance_map_integrand, [[0, f * theta_d], [-np.pi, np.pi]], [f, gamma, fsr, rms, phase_dev])[0]
        return integral / (np.pi * f**2 * theta_d**2)


class P1(Pure):

    def __init__(self, fopd, theta_t, gamma_m, gamma_a, lam, t, t_ref, p):
        """

        :param fopd:
        :param theta_t:
        :param gamma_m:
        :param gamma_a:
        :param lam:
        :param t:
        :param t_ref:
        :param p:
        """
        Pure.__init__(self, fopd, theta_t, gamma_m, gamma_a, lam, t_ref, p)
        self.t = t
        self.glass = P_SF68()
        self.n_ref_glass = self.glass.sellmeier_1(self.lam)
        self.n_glass = self.generate_n_glass(self.t)

        self.air = Air()
        self.n_air = self.generate_n_air(self.t)

        self.d_glass, self.d_air = self.generate_thickness()

    def generate_n_glass(self, t):
        """

        :param t:
        :return:
        """
        return self.n_ref_glass + self.glass.delta_n(t, self.n_ref_glass, self.lam)

    def generate_n_air(self, t):
        """

        :param t:
        :return:
        """
        return self.air.n_air(self.lam, t, self.p)

    def generate_thickness(self):
        """
        Calculate the thicknesses of the three materials.

        Uses Eq 2, 6, and 7 from [1].
        :return: Tuple. The thicknesses (d1, d2, d3).
        """
        root1 = np.sqrt(self.n_glass ** 2 - np.sin(self.theta_t) ** 2)
        root2 = np.sqrt(self.n_air ** 2 - np.sin(self.theta_t) ** 2)

        m1 = 2 * np.asarray([[self.n_glass * np.sqrt(1 - (np.sin(self.theta_t) / self.n_glass) ** 2), -self.n_air * np.sqrt(1 - (np.sin(self.theta_t) / self.n_air) ** 2)],
                          [-1 / (2 * root1), 1 / (2 * root2)]])
        v1 = np.asarray([self.fopd, 0])

        return tuple(la.solve(m1, v1))

    def d_glass_thermal_expansion(self, t):
        """
        Calculate the thermal expansion of the pure glass arm.

        Because d1 was determined with self.t, the reference temperature used here is self.t, not self.pure.t_ref.
        :param t: Float. The temperature at which to calculate the thermal expansion.
        :return: Float. The thermally expanded length of the glass.
        """
        return self.d_glass * (1 + self.glass.a * (t - self.t))

    def local_transmittance(self, d_phi, gamma, fsr, rms=0, phase_dev=0):
        """
        Calculate the local transmittance.

        For a given FWMI and incident beam, the output is a function of the angle in the plane of the FWMI.
        The incident intensities are assumed to both be 1.
        Eq. 14 in [1].

        The rms wavefront error can be incorporated for small errors by setting the rms argument.
        The wavefront error has a general form for different sources of error, see section 3.2.2c and Eq. 23 of [1].
        :param d_phi: Float. The phase difference between a ray incident at some angle and a ray incident at the tilt angle.
        :param gamma: Float. The linewidth of the signal (molecular or aerosol).
        :param fsr: Float. The FSR of the FWMI. A function of the tilt angle.
        :param rms: Float. The rms wavefront error. Default is 0.
        :param phase_dev: Float. The frequency shift from a perfect lock.
        :return: Float. The local transmittance.
        """
        rms = 2 * np.pi * self.nu * rms / const.c
        return 1 / 2 - 1 / 2 * np.exp(-(np.pi * gamma / fsr)**2) * np.cos(d_phi + phase_dev * 2 * np.pi / fsr) * (1 - rms**2 / 2)

    def mapping_angle(self, rho, phi, f):
        """
        Calculate the angle used in the transmittance map function.

        The map is from the exit angle of the FWMI to the focal plane of the exit lens, Eq. 16 in [1].
        :param rho: Float. The radial coordinate in the exit lens's image plane.
        :param phi: Float. The azimuthal coordinate in the exit lens's image plane.
        :param theta_t: Float. The tilt angle.
        :param f: Float. The exit lens's focal length.
        :return: Float. The angle to use with the local transmittance mapping function.
        """
        return np.arccos((2 * f * np.cos(self.theta_t)**2 - rho * np.sin(2 * self.theta_t) * np.cos(phi)) / (2 * np.sqrt(f**2 + rho**2) * np.cos(self.theta_t)))

    def transmittance_map_integrand(self, rho, phi, f, gamma, fsr, rms=0, phase_dev=False):
        """
        Calculate the integrand of the overall transmittance function.

        Based on the map from the local transmittance to the transmittance at the exit lens image plane.
        :param rho: Float. The radial coordinate in the exit lens's image plane.
        :param phi: Float. The azimuthal coordinate in the exit lens's image plane.
        :param f: Float. The exit lens's focal length.
        :param gamma: Float. The linewidth of the signal (molecular or aerosol).
        :param fsr: Float. The FSR of the FWMI. A function of the tilt angle.
        :param rms: Float. The rms wavefront error. Default is 0.
        :param phase_dev: Float. The frequency shift from a perfect lock.
        :return: Float. The integrand of the overall transmittance.
        """
        theta = self.mapping_angle(rho, phi, f)
        opd_theta = self.opd_exact_pure(theta, self.n_glass, self.d_glass, self.n_air, self.d_air)
        del_phi = self.delta_phi(opd_theta, const.c / fsr)  # const.c / fsr instead of fopd to make changing fopd easier
        return rho * self.local_transmittance(del_phi, gamma, fsr, rms, phase_dev)

    def overall_transmittance(self, theta_d, f, gamma, fsr, rms=0, phase_dev=False):
        """
        Calculate the overall transmittance in the focal plane of the exit lens.

        Eq. 15 in [1].
        :param theta_d: Float. The HALF divergent angle that sets the limit on the integrand.
        :param f: Float. The exit lens's focal length.
        :param gamma: Float. The linewidth of the signal (molecular or aerosol).
        :param fsr: Float. The FSR of the FWMI. A function of the tilt angle.
        :param rms: Float. The rms wavefront error. Default is 0.
        :param phase_dev: Float. The frequency shift from a perfect lock.
        :return:
        """
        integral = nquad(self.transmittance_map_integrand, [[0, f * theta_d], [-np.pi, np.pi]], [f, gamma, fsr, rms, phase_dev])[0]
        return integral / (np.pi * f**2 * theta_d**2)


class Copper(Pure):

    def __init__(self, fopd, theta_t, gamma_m, gamma_a, lam, t, t_ref, p, d_opd_d_t=0, glass=N_SF66):
        """

        :param fopd:
        :param theta_t:
        :param gamma_m:
        :param gamma_a:
        :param lam:
        :param t:
        :param t_ref:
        :param p:
        """
        Pure.__init__(self, fopd, theta_t, gamma_m, gamma_a, lam, t_ref, p)
        self.t = t
        self.d_opd_d_t = d_opd_d_t
        self.glass = glass()
        self.n_ref_glass = self.glass.sellmeier_1(self.lam)
        self.n_glass = self.generate_n_glass(self.t)

        self.air = Air()
        self.n_air = self.generate_n_air(self.t)

        self.d_glass, self.d_air = self.generate_thickness()
        self.copper_a = 17e-6  # coefficient of thermal expansion for copper
        # self.d_copper = 0 * d_opd_d_t / abs(self.glass.a - self.copper_a)  # Thickness of copper
        # self.d_copper = self.n_glass / self.n_air * self.d_glass * self.glass.a / self.copper_a  #(d_opd_d_t + self.d_glass * self.glass.a) / self.copper_a
        if d_opd_d_t == None:
            self.d_copper = self.d_air
        else:
            self.d_copper = self.generate_copper_thickness(d_opd_d_t)
        print(self.d_copper, self.d_air, self.d_glass, self.d_air * self.n_air, self.d_glass * self.n_glass)

    def generate_n_glass(self, t):
        """

        :param t:
        :return:
        """
        return self.n_ref_glass + self.glass.delta_n(t, self.n_ref_glass, self.lam)

    def generate_n_air(self, t):
        """

        :param t:
        :return:
        """
        return self.air.n_air(self.lam, t, self.p)

    def generate_thickness(self):
        """
        Calculate the thicknesses of the three materials.

        Uses Eq 2, 6 from [1].
        :return: Tuple. The thicknesses (d1, d2, d3).
        """
        root1 = np.sqrt(self.n_glass ** 2 - np.sin(self.theta_t) ** 2)
        root2 = np.sqrt(self.n_air ** 2 - np.sin(self.theta_t) ** 2)

        m1 = 2 * np.asarray([[self.n_glass * np.sqrt(1 - (np.sin(self.theta_t) / self.n_glass) ** 2), -self.n_air * np.sqrt(1 - (np.sin(self.theta_t) / self.n_air) ** 2)],
                          [-1 / (2 * root1), 1 / (2 * root2)]])
        v1 = np.asarray([self.fopd, 0])

        return tuple(la.solve(m1, v1))

    def generate_copper_thickness(self, d_opd_dt):
        """
        Calculate the thickness of the copper spacer required to produce a particular OPD temperature tuning

        Calculated using Eq. 7 from [1]. In this case, there is only material 1 and 2, and the alpha_2 * d_2 term
        is alpha_copper * d_copper.
        :param d_opd_dt:
        :return:
        """
        root1 = np.sqrt(self.n_glass ** 2 - np.sin(self.theta_t) ** 2)
        root2 = np.sqrt(self.n_air ** 2 - np.sin(self.theta_t) ** 2)
        b1 = self.glass.beta(self.t, self.n_ref_glass, self.lam)
        b2 = self.air.beta_air(self.lam, self.t, self.p)
        coef = -1 / (self.copper_a * root2)
        term1 = self.glass.a * self.d_glass * root1
        term2 = b1 * self.n_glass * self.d_glass / root1
        term3 = b2 * self.n_air * self.d_air / root2
        return coef * (d_opd_dt / 2 - term1 - term2 + term3)

    def d_glass_thermal_expansion(self, t):
        """
        Calculate the thermal expansion of the pure glass arm.

        Because d1 was determined with self.t, the reference temperature used here is self.t, not self.pure.t_ref.
        :param t: Float. The temperature at which to calculate the thermal expansion.
        :return: Float. The thermally expanded length of the glass.
        """
        return self.d_glass * (1 + self.glass.a * (t - self.t))

    def d_air_thermal_expansion(self, t):
        """
        Calculate the thermal expansion of the pure glass arm.

        Because d1 was determined with self.t, the reference temperature used here is self.t, not self.pure.t_ref.
        :param t: Float. The temperature at which to calculate the thermal expansion.
        :return: Float. The thermally expanded length of the glass.
        """
        return self.d_air + self.d_copper * (self.copper_a * (t - self.t))
        # return self.d_air * (1 + self.copper_a * (t - self.t))

    def local_transmittance(self, d_phi, gamma, fsr, rms=0, phase_dev=0):
        """
        Calculate the local transmittance.

        For a given FWMI and incident beam, the output is a function of the angle in the plane of the FWMI.
        The incident intensities are assumed to both be 1.
        Eq. 14 in [1].

        The rms wavefront error can be incorporated for small errors by setting the rms argument.
        The wavefront error has a general form for different sources of error, see section 3.2.2c and Eq. 23 of [1].
        :param d_phi: Float. The phase difference between a ray incident at some angle and a ray incident at the tilt angle.
        :param gamma: Float. The linewidth of the signal (molecular or aerosol).
        :param fsr: Float. The FSR of the FWMI. A function of the tilt angle.
        :param rms: Float. The rms wavefront error. Default is 0.
        :param phase_dev: Float. The frequency shift from a perfect lock.
        :return: Float. The local transmittance.
        """
        rms = 2 * np.pi * self.nu * rms / const.c
        return 1 / 2 - 1 / 2 * np.exp(-(np.pi * gamma / fsr)**2) * np.cos(d_phi + phase_dev * 2 * np.pi / fsr) * (1 - rms**2 / 2)

    def mapping_angle(self, rho, phi, f):
        """
        Calculate the angle used in the transmittance map function.

        The map is from the exit angle of the FWMI to the focal plane of the exit lens, Eq. 16 in [1].
        :param rho: Float. The radial coordinate in the exit lens's image plane.
        :param phi: Float. The azimuthal coordinate in the exit lens's image plane.
        :param theta_t: Float. The tilt angle.
        :param f: Float. The exit lens's focal length.
        :return: Float. The angle to use with the local transmittance mapping function.
        """
        return np.arccos((2 * f * np.cos(self.theta_t)**2 - rho * np.sin(2 * self.theta_t) * np.cos(phi)) / (2 * np.sqrt(f**2 + rho**2) * np.cos(self.theta_t)))

    def transmittance_map_integrand(self, rho, phi, f, gamma, fsr, rms=0, phase_dev=False):
        """
        Calculate the integrand of the overall transmittance function.

        Based on the map from the local transmittance to the transmittance at the exit lens image plane.
        :param rho: Float. The radial coordinate in the exit lens's image plane.
        :param phi: Float. The azimuthal coordinate in the exit lens's image plane.
        :param f: Float. The exit lens's focal length.
        :param gamma: Float. The linewidth of the signal (molecular or aerosol).
        :param fsr: Float. The FSR of the FWMI. A function of the tilt angle.
        :param rms: Float. The rms wavefront error. Default is 0.
        :param phase_dev: Float. The frequency shift from a perfect lock.
        :return: Float. The integrand of the overall transmittance.
        """
        theta = self.mapping_angle(rho, phi, f)
        opd_theta = self.opd_exact_pure(theta, self.n_glass, self.d_glass, self.n_air, self.d_air)
        del_phi = self.delta_phi(opd_theta, const.c / fsr)  # const.c / fsr instead of fopd to make changing fopd easier
        return rho * self.local_transmittance(del_phi, gamma, fsr, rms, phase_dev)

    def overall_transmittance(self, theta_d, f, gamma, fsr, rms=0, phase_dev=False):
        """
        Calculate the overall transmittance in the focal plane of the exit lens.

        Eq. 15 in [1].
        :param theta_d: Float. The HALF divergent angle that sets the limit on the integrand.
        :param f: Float. The exit lens's focal length.
        :param gamma: Float. The linewidth of the signal (molecular or aerosol).
        :param fsr: Float. The FSR of the FWMI. A function of the tilt angle.
        :param rms: Float. The rms wavefront error. Default is 0.
        :param phase_dev: Float. The frequency shift from a perfect lock.
        :return:
        """
        integral = nquad(self.transmittance_map_integrand, [[0, f * theta_d], [-np.pi, np.pi]], [f, gamma, fsr, rms, phase_dev])[0]
        return integral / (np.pi * f**2 * theta_d**2)



if __name__ == "__main__":
    configuration = "hybrid"
    if configuration == "pure":
        switch_lam = 2
        lam = 532e-9 * switch_lam
        theta_t = 1.5 * np.pi / 180
        gamma_m = 1.40e9 / switch_lam  # molecular signal spectral width
        gamma_a = 50e6 / switch_lam  # aerosol signal spectral width
        fopd = 0.1 * switch_lam
        t_ref = 20
        t = 20
        p = 1
        f = 0.1
        # fopd = 0.1000000255239908
        p1 = P1(fopd, theta_t, gamma_m, gamma_a, lam, t, t_ref, p)
        opd = p1.opd_exact_pure(theta_t, p1.n_glass, p1.d_glass, p1.n_air, p1.d_air)
        opd2 = p1.opd_exact_pure(theta_t,
                            p1.generate_n_glass(t),
                            p1.d_glass_thermal_expansion(t),
                            p1.generate_n_air(t),
                            p1.d_air)
        d_glass = p1.d_glass
        d_air = p1.d_air
        print(p1.d_glass, p1.d_air, opd, opd2, (opd2 - fopd) / lam)

        d_glass_paper = 0.0327670
        d_air_paper = 0.0162143
        n_ref_glass_paper = 2.0209909911203465

        d_d_glass = d_glass_paper - d_glass
        d_d_air = d_air_paper - d_air

        print(d_d_glass, d_d_air)
    elif configuration == "hybrid":
        switch_lam = 2
        lam = 532e-9 * switch_lam
        theta_t = 1.5 * np.pi / 180
        gamma_m = 1.40e9 / switch_lam  # molecular signal spectral width
        gamma_a = 50e6 / switch_lam  # aerosol signal spectral width
        fopd = 0.1 * switch_lam
        t_ref = 20
        t = 20
        p = 1
        f = 0.1
        # fopd = 0.1000000255239908
        h1 = H1(fopd, theta_t, gamma_m, gamma_a, lam, t, t_ref, p)
        opd = h1.opd_exact(theta_t, h1.n1, h1.d1, h1.n2, h1.d2, h1.n3, h1.d3)
        opd2 = h1.opd_exact(theta_t,
                            h1.generate_n1(t),
                            h1.d1_thermal_expansion(t),
                            h1.generate_n2(t),
                            h1.d2_thermal_expansion(t),
                            h1.generate_n3(t),
                            h1.d3)
        d1 = h1.d1
        d2 = h1.d2
        d3 = h1.d3
        print(h1.d1, h1.d2, h1.d3, opd, opd2, (opd2 - fopd) / lam)

        d1_paper = 0.053220
        d2_paper = 0.0167970
        d3_paper = 0.0191608
        n_ref_1_paper = 1.9374328041896338
        n_ref_2_paper = 2.0209909911203465

        d_d1 = d1_paper - d1
        d_d2 = d2_paper - d2
        d_d3 = d3_paper - d3

        print(d_d1, d_d2, d_d3)
