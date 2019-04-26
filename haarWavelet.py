"""Utilities for analysis and synthesis of 1d and 2d signals/images with the Haar wavelet basis. Major restriction: Data must be of size 2**J (1d) or 2**J * 2**J (2d)."""

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log, ceil, log10
def unpackWavelet(waco):
	"""Take a "packed" 2d wavelet coefficient list and flatten it to a single 1d list, then return it."""

	J = len(waco)
	unpacked = np.zeros((2**(2*(J-1)),)) ##### !!!!!
	unpacked[0] = waco[0][0,0]
	for j in range(1, J):
		unpacked[2**(2*j-2):2**(2*j)] = np.concatenate((waco[j][0].flatten(), waco[j][1].flatten(), waco[j][2].flatten()))
	return unpacked

def packWavelet(vector):
	"""Take a list of length 2**J and "fan" it into a 2d packed wavelet coefficient list, then return it."""
	
	packed = [np.array([[vector[0]]])]
	J = int(log10(len(vector))/(2*log10(2)))+1
	for j in range(1, J):
		temp1 = np.reshape(vector[2**(2*j-2):2**(2*j-1)], (2**(j-1), 2**(j-1)))
		temp2 = np.reshape(vector[2**(2*j-1):2**(2*j-1)+2**(2*j-2)], (2**(j-1), 2**(j-1)))
		temp3 = np.reshape(vector[2**(2*j-1)+2**(2*j-2):2**(2*j)], (2**(j-1), 2**(j-1)))
		packed.append([temp1, temp2, temp3])
	return packed

def checkWhether2dWaveletCoeff(coeff):
	"""Check whether coeff is indeed a valid 2d wavelet coefficient list and return this as a boolean."""

	N = len(coeff)
	if not isinstance(coeff, list):
		return False
	a0 = coeff[0]
	if not (isinstance(a0, np.ndarray) and a0.ndim == 2):
		return False
	for n, a in enumerate(coeff[1:]):
		if not (isinstance(a, list) and len(a) == 3):
			return False
		for amat in a:
			if not (isinstance(amat, np.ndarray) and amat.shape == (2**n, 2**n)):
				return False
	return True

# ========== 1d Wavelet tools =============
def waveletanalysis(f):
	"""Calculate and return the Haar wavelet decomposition of a list of length 2**J."""

	a = [f]
	d = [0]
	J = int(log(len(f), 2)) # maximal resolution
	for j in range(J):
		a_last = a[-1]
		a.append((a_last[0::2] + a_last[1::2])/sqrt(2))
		d.append((-a_last[0::2] + a_last[1::2])/sqrt(2))
	
	w = [a[-1]/(2**(J/2))] # adjust for absolute size
	for j in range(J):
		#w.append(d[J-j])
		w.append(d[J-j]/(2**(J/2))) # adjust for absolute size
	return w



def waveletsynthesis(w,resol=None):
	"""Take a wavelet coefficient vector and calculate the represented signal.
	
	Optionally, you can supply a custom resolution resol to cast the result in. If resol is lower than the intrinsic resolution of the coefficient vector, this decreases the resolution of the result. If resol is higher, this leads to padding with piecewise constant continuations of the signal.
	"""

	if resol is None:
		J = len(w) - 1
	else:
		J = resol

	
	f = np.zeros((2**J,)) + w[0]
	for j in range(1, min(J+1, len(w))):
		for k, c in enumerate(w[j]):
			psivec = np.zeros((2**J,))
			psivec[2**(J-j+1)*k:2**(J-j+1)*k + 2**(J-j)] = -2**((j-1)/2)
			psivec[2**(J-j+1)*k + 2**(J-j):2**(J-j+1)*(k+1)] = 2**((j-1)/2)
			f = f + c*psivec
	return f


def unpackWavelet_1d(waco):
	"""Take a "packed" 1d wavelet coefficient list and flatten it to a single 1d list, then return it."""

	J = len(waco)
	unpacked = np.zeros((2**(J-1)))
	unpacked[0] = waco[0]
	for j in range(1, J):
		unpacked[2**(j-1):2**(j)] = waco[j]
	return unpacked

# ========== 2d Wavelet tools =============

def waveletsynthesis2d(w, resol=None):
	"""Take a wavelet coefficient vector and calculate the represented signal.
		
		Optionally, you can supply a custom resolution resol to cast the result in. If resol is lower than the intrinsic resolution of the coefficient vector, this decreases the resolution of the result. If resol is higher, this leads to padding with piecewise constant continuations of the signal.
		"""
	if resol is None:
		J = len(w) - 1
	else:
		J = max(resol, len(w) - 1)
	f = np.zeros((2**J, 2**J))+ w[0]
	for j in range(1, len(w)):
		w_hori = w[j][0] # is quadratic
		w_vert = w[j][1]
		w_diag = w[j][2]
		(maxK, maxL) = w_hori.shape
		for k in range(maxK):
			for l in range(maxL):
				psivec1 = np.zeros((2**J, 2**J))
				psivec1[2**(J-j+1)*k:2**(J-j+1)*k + 2**(J-j), 2**(J-j+1)*l:2**(J-j+1)*(l+1)] = 2**(j-1)
				psivec1[2**(J-j+1)*k + 2**(J-j):2**(J-j+1)*(k+1), 2**(J-j+1)*l:2**(J-j+1)*(l+1)] = -2**(j-1)
				psivec2 = np.zeros((2**J, 2**J))
				psivec2[2**(J-j+1)*k:2**(J-j+1)*(k+1), 2**(J-j+1)*l:2**(J-j+1)*l + 2**(J-j)] = 2**(j-1)
				psivec2[2**(J-j+1)*k:2**(J-j+1)*(k+1), 2**(J-j+1)*l + 2**(J-j):2**(J-j+1)*(l+1)] = -2**(j-1)
				psivec3 = np.zeros((2**J, 2**J))
				psivec3[2**(J-j+1)*k:2**(J-j+1)*k + 2**(J-j), 2**(J-j+1)*l:2**(J-j+1)*l + 2**(J-j)] = 2**(j-1)
				psivec3[2**(J-j+1)*k + 2**(J-j):2**(J-j+1)*(k+1), 2**(J-j+1)*l:2**(J-j+1)*l + 2**(J-j)] = -2**(j-1)
				psivec3[2**(J-j+1)*k:2**(J-j+1)*k + 2**(J-j), 2**(J-j+1)*l + 2**(J-j):2**(J-j+1)*(l+1)] = -2**(j-1)
				psivec3[2**(J-j+1)*k + 2**(J-j):2**(J-j+1)*(k+1), 2**(J-j+1)*l + 2**(J-j):2**(J-j+1)*(l+1)] = 2**(j-1)
				f = f + w_hori[k,l]*psivec1 + w_vert[k, l]*psivec2 + w_diag[k,l]*psivec3
	return f

def waveletanalysis2d(f):
	"""Calculate and return the Haar wavelet decomposition of an array of size 2**J * 2**J."""
	a = [f]
	d = [0]		
	J = int(log(f.shape[0], 2))
	for j in range(J):
		a_last = a[-1]
		temp1 = (a_last[0::2, :] + a_last[1::2, :])/2
		a.append((temp1[:, 0::2] + temp1[:, 1::2])/2)
		
		temp2 = (a_last[0::2, :] - a_last[1::2, :])/2
		d1 = (temp2[:, 0::2] + temp2[:, 1::2])/(2**(J-j))
		d2 = (temp1[:, 0::2] - temp1[:, 1::2])/(2**(J-j))
		d3 = (temp2[:, 0::2] - temp2[:, 1::2])/(2**(J-j))
		d.append([d1,d2,d3])
	w = [a[-1]]
	for j in range(J):
		w.append(d[J-j])
	return w

		




def parseResolution(coeffs, newRes):
	"""Parse a Wavelet decomposition to another resolution (by padding with zero coefficients or dropping some)."""
	newCoeffs = packWavelet(np.copy(unpackWavelet(coeffs)))
	res = len(coeffs)
	if newRes == res:
		# nothing to be done
		return newCoeffs
	if newRes < res:
		return newCoeffs[0:newRes]
	if newRes > res:
		for k in range(res, newRes):
			c1 = np.zeros((2**(k-1), 2**(k-1)))
			c2 = np.zeros((2**(k-1), 2**(k-1)))
			c3 = np.zeros((2**(k-1), 2**(k-1)))
			newCoeffs.append([c1, c2, c3])
		return newCoeffs
			


if __name__ == "__main__":
	J = 9
	num = 2**J
	x = np.linspace(0, 1, 2**(J), endpoint=False)
	gg1 = lambda x: 1 + 2**(-J)/(x**2+2**J) + 2**J/(x**2 + 2**J)*np.cos(32*x)
	g1 = lambda x: gg1(2**J*x)
	gg2 = lambda x: (1 - 0.4*x**2)/(2**(J+3)) + np.sin(7*x/(2*pi))/(1 + x**2/2**J)
	g2 = lambda x: gg2(2**J*x)
	gg3 = lambda x: 3 + 3*(x**2/(2**(2*J)))*np.sin(x/(8*pi))
	g3 = lambda x: gg3(2**J*x)
	gg4 = lambda x: (x**2/3**J)*0.1*np.cos(x/(2*pi))-x**3/8**J + 0.1*np.sin(3*x/(2*pi))
	g4 = lambda x: gg4(2**J*x)
	

	vec1 = g2(x[0:2**(J-2)])
	vec2 = g1(x[2**(J-2):2**(J-1)])
	vec3 = g3(x[2**(J-1):2**(J)-2**(J-1)])
	vec4 = g4(x[2**(J)-2**(J-1):2**(J)])

	f = np.concatenate((vec1, vec2, vec3, vec4))

	w = waveletanalysis(f)

	plt.figure()
	titles = ["Undersampling", "Undersampling", "exact sampling", "Oversampling"]
	for n, resol in enumerate(np.arange(7, 11)):
		plt.subplot(4,1,n+1)
		xx = np.linspace(0, 1, 2**resol, endpoint=False)
		ff = waveletsynthesis(w, resol=resol)
		plt.plot(x, f, 'g--')
		plt.plot(xx, ff)
		plt.title(titles[n])
		
	

	
	plt.ion()
	plt.show()
	
	
	J = 2**7
	X = np.linspace(-5, 5, J)
	Y = np.linspace(-5, 5, J)
	X, Y = np.meshgrid(X, Y)
	R = np.sqrt(X**4 + Y**2 + X**2*Y**2)
	Z = np.sin(R)*np.cos(1/5*(X-Y**2))*np.sin(Y-np.exp(X))
	
	hwa = waveletanalysis2d(Z)
	B = waveletsynthesis2d(hwa)
	
	plt.figure()
	plt.subplot(3, 3, 1)
	plt.imshow(Z, cmap=plt.cm.coolwarm, interpolation='none')
	for k in range(1, len(hwa)+1):
		wc = parseResolution(hwa, k)
		plt.subplot(3,3,k+1)
		plt.imshow(waveletsynthesis2d(wc), cmap=plt.cm.coolwarm, interpolation='none')
	
	
	

