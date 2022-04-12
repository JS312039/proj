# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 20:38:59 2022

@author: Julia
"""

from math import sin, cos, sqrt, atan, atan2, degrees, radians, tan, pi, acos, floor
from numpy import array, mean, genfromtxt, savetxt, transpose, zeros

class Transformacje:
    def __init__(self, model: str = "wgs84"):
        """
        Parametry elipsoid:
            a - duża półoś elipsoidy - promień równikowy
            b - mała półoś elipsoidy - promień południkowy
            flat - spłaszczenie
            ecc2 - mimośród^2
        + WGS84: https://en.wikipedia.org/wiki/World_Geodetic_System#WGS84
        + Inne powierzchnie odniesienia: https://en.wikibooks.org/wiki/PROJ.4#Spheroid
        + Parametry planet: https://nssdc.gsfc.nasa.gov/planetary/factsheet/index.html
        """
        if model == "wgs84":
            self.a = 6378137.0 # semimajor_axis
            self.b = 6356752.31424518 # semiminor_axis
        elif model == "grs80":
            self.a = 6378137.0
            self.b = 6356752.31414036
        elif model == "mars":
            self.a = 3396900.0
            self.b = 3376097.80585952
        else:
            raise NotImplementedError(f"{model} model not implemented")
        self.flat = (self.a - self.b) / self.a
        self.ecc = sqrt(2 * self.flat - self.flat ** 2) # eccentricity  WGS84:0.0818191910428 
        self.ecc2 = (2 * self.flat - self.flat ** 2) # eccentricity**2

    
    
    def xyz2plh(self, X, Y, Z, output = 'dec_degree'):
        """
        Algorytm Hirvonena - algorytm transformacji współrzędnych ortokartezjańskich (x, y, z)
        na współrzędne geodezyjne długość szerokość i wysokośc elipsoidalna (phi, lam, h). Jest to proces iteracyjny. 
        W wyniku 3-4-krotneej iteracji wyznaczenia wsp. phi można przeliczyć współrzędne z dokładnoscią ok 1 cm.     
        Parameters
        ----------
        X, Y, Z : [float] - współrzędne w układzie orto-kartezjańskim

        Returns
        -------
        lat: [stopnie dziesiętne] - szerokość geodezyjna
        lon: [stopnie dziesiętne] - długośc geodezyjna.
        h : [metry] - wysokość elipsoidalna
        output [STR] - optional, defoulf 
            dec_degree - decimal degree
            dms - degree, minutes, sec
        """
        r   = sqrt(X**2 + Y**2)           # promień
        lat_prev = atan(Z / (r * (1 - self.ecc2)))    # pierwsze przybliilizenie
        lat = 0
        while abs(lat_prev - lat) > 0.000001/206265:    
            lat_prev = lat
            N = self.a / sqrt(1 - self.ecc2 * sin(lat_prev)**2)
            h = r / cos(lat_prev) - N
            lat = atan((Z/r) * (((1 - self.ecc2 * N/(N + h))**(-1))))
        lon = atan(Y/X)
        N = self.a / sqrt(1 - self.ecc2 * (sin(lat))**2);
        h = r / cos(lat) - N       
        if output == "dec_degree":
            return degrees(lat), degrees(lon), h 
        elif output == "dms":
            lat = self.deg2dms(degrees(lat))
            lon = self.deg2dms(degrees(lon))
            return f"{lat[0]:02d}:{lat[1]:02d}:{lat[2]:.2f}", f"{lon[0]:02d}:{lon[1]:02d}:{lon[2]:.2f}", f"{h:.3f}"
        else:
            raise NotImplementedError(f"{output} - output format not defined")
            
    def dms(degrees):
        """
        Funkcja przeliczająca radiany na stopnie, minuty i sekundy
        Parameters
        ----------
        degrees: [float][stopnie dziesiętne] - wartosć w stopniach dziesiętnych
        
        Returns
        -------
        wynik: [list] - lista z trzema wartosciami [stopnie, minuty, sekundy]
        """
        if degrees >= 0:
            d = floor(degrees)
            mi = floor((degrees - d) * (60))
            s = (((degrees - d) * (60)) - mi) * (60)
        else:
            degrees *= -1
            d = floor(degrees)
            mi = floor((degrees - d) * (60))
            s = (((degrees - d) * (60)) - mi) * (60)
            d *= -1
            s *= -1
            mi *= -1
        return [d,mi,round(s,5)]
    
    def flh2xyz(self,phi,lam,h):
        """
        Algorytm odwrotny - algorytm transformacji współrzędnych geodezyjnych: 
        długość szerokość i wysokość elipsoidalna (phi, lam, h) 
        na współrzędne ortokartezjańskie (x, y, z). 
        
        Parameters
        ----------
        phi: [float][stopnie dziesiętne] - szerokość geodezyjna
        lam: [float][stopnie dziesiętne] - długość geodezyjna
        h: [metry] - wysokość elipsoidalna
        
        Returns
        -------
        X, Y, Z: [float] - współrzędne w układzie orto-kartezjańskim
        """
        phi=radians(phi)
        lam=radians(lam)
        N = self.a/(1-self.ecc2*(sin(phi))**2)**(0.5)
        X= (N+ h) * cos(phi) * cos(lam)
        Y= (N+ h) * cos(phi) * sin(lam)
        Z= (N*(1-self.ecc2)+h) * sin(phi) 
        return f"{X:.3f}", f"{Y:.3f}",f"{Z:.3f}"
    
    
    def u2000(self, phi, lam):
        """
        Przeliczenie współrzędnych geodezyjnych fi i lambda do współrzędnych Układu 2000 
    
        INPUT:
            phi : [float][stopnie dziesiętne] - szerokość geodezyjna 
            lam : [float][stopnie dziesiętne] - długość geodezyjna 
        
        OUTPUT:
            x00 :[float] : współrzędna w układzie lokalnym 2000 (metry)
            y00 :[float] : współrzędna w układzie lokalnym 2000 (metry)
    
        """
        phi=radians(phi)
        lam=radians(lam)
        m = 0.999923
        N = self.a/(1-self.ecc2*(sin(phi))**2)**(0.5)
        t = tan(phi)
        e_2 = self.ecc2/(1-self.ecc2)
        n2 = e_2 * (cos(phi))**2    
        lam = degrees(lam)
        # if lam>13.5 and lam <16.5:
        #     s = 5
        #     lam0 = 15
        # elif lam>16.5 and lam <19.5:
        #     s = 6
        #     lam0 = 18
        # elif lam>19.5 and lam <22.5:
        #     s = 7
        #     lam0 = 21
        # elif lam>22.5 and lam <25.5:
        #     s = 8
        #     lam0 = 24 
        lam0 = 21
        s = 7
        lam = radians(lam)
        lam0 = radians(lam0)
        l = lam - lam0
        A0 = 1 - (self.ecc2/4) - ((3*(self.ecc2**2))/64) - ((5*(self.ecc2**3))/256)   
        A2 = (3/8) * (self.ecc2 + ((self.ecc2**2)/4) + ((15 * (self.ecc2**3))/128))
        A4 = (15/256) * (self.ecc2**2 + ((3*(self.ecc2**3))/4))
        A6 = (35 * (self.ecc2**3))/3072 
        sig = self.a * ((A0*phi) - (A2*sin(2*phi)) + (A4*sin(4*phi)) - (A6*sin(6*phi))) 
        x = sig + ((l**2)/2) * N *sin(phi) * cos(phi) * (1 + ((l**2)/12) * ((cos(phi))**2) * (5 - t**2 + 9*n2 + 4*(n2**2)) + ((l**4)/360) * ((cos(phi))**4) * (61 - (58*(t**2)) + (t**4) + (270*n2) - (330 * n2 *(t**2))))
        y = l * (N*cos(phi)) * (1 + ((((l**2)/6) * (cos(phi))**2) * (1-t**2+n2)) +  (((l**4)/(120)) * (cos(phi)**4)) * (5 - (18 * (t**2)) + (t**4) + (14*n2) - (58*n2*(t**2))))
        x00 = round(m * x, 3) 
        y00 = round(m * y + (s*1000000) + 500000, 3)
        return f"{x00:.3f}", f"{y00:.3f}"

    def u1992(self, phi,lam):
        """
        Przeliczenie współrzędnych geodezyjnych fi i lambda do współrzędnych Układu 1992 
    
        INPUT:
            phi : [float][stopnie dziesiętne] - szerokość geodezyjna 
            lam : [float][stopnie dziesiętne] - długość geodezyjna 
        
        OUTPUT:
            x92 :[float] : współrzędna w układzie lokalnym 1992 (metry)
            y92 :[float] : współrzędna w układzie lokalnym 1992 (metry)
    
        """
        phi = radians(phi)
        lam = radians(lam)
        L0 = radians(19)
        e2_ = (self.a**2 - self.b**2)/(self.b**2)
        eta2 = e2_ * cos(phi)**2
        t = tan(phi)
        l = lam - L0
        N = self.a/(1-self.ecc2*(sin(phi))**2)**(0.5)
        A0 = 1 - (self.ecc2/4) - ((3*(self.ecc2**2))/64) - ((5*(self.ecc2**3))/256)   
        A2 = (3/8) * (self.ecc2 + ((self.ecc2**2)/4) + ((15 * (self.ecc2**3))/128))
        A4 = (15/256) * (self.ecc2**2 + ((3*(self.ecc2**3))/4))
        A6 = (35 * (self.ecc2**3))/3072 
        sig = self.a * ((A0*phi) - (A2*sin(2*phi)) + (A4*sin(4*phi)) - (A6*sin(6*phi)))
        x = sig + (l**2)/2 * N * sin(phi) * cos(phi) * ( 1+(((l**2)/12) * (cos(phi)**2) * (5 - t**2 + 9*eta2 + 4*(eta2**2)) ) + (((l**4)/360) * (cos(phi)**4) * (61 - 58*(t**2) + (t**4) + 270*eta2 - 330*(eta2)*(t**2)) ) )
        y = l * N * cos(phi) * ( 1+(((l**2)/6) * (cos(phi)**2) * (1 - t**2 + eta2) ) + (((l**4)/120) * (cos(phi)**4) * (5 - 18*(t**2) + (t**4) + 14*eta2 - 58*(eta2)*(t**2)) ) )
        m0 = 0.9993
        x92 = m0*x - 5300000
        y92 = m0*y + 500000
        return f"{x92:.3f}", f"{y92:.3f}"
        

    
    def neu(self,x, y, z):
        """ 
        Funkcja przeliczająca współrzędne ortokartezjańskie (x,y,z) na NEU 
        
        INPUT:
            x, y, z : [float] - współrzędne w układzie orto-kartezjańskim
        
        OUTPUT:
            n :[float] : współrzędna topocentryczna (ENU)
            e :[float] : współrzędna topocentryczna (ENU)
            u :[float] : współrzędna topocentryczna (ENU)
        """
        x0 = mean(X)
        y0 = mean(Y)
        z0 = mean(Z)
        b0, l0, h0 = geo.xyz2plh(x0, y0, z0)
        b0, l0 = radians(b0), radians(l0)
        R = array([
            [-sin(l0), -cos(l0) *sin(b0), cos(b0) * cos(l0)],
            [cos(l0), -sin(l0) * sin(b0), sin(l0) * cos(b0)],
            [0, cos(b0), sin(b0)]])
        n, e, u = R.dot(array([x0 - x, y0 - y, z0 - z]).T)
        return n, e, u
    
    def azimuth_dist_xy(xA, yA, xB, yB):
        """
        Wyznaczenie azymutu AB i odległości skośnej pomiędzy punktami AB
        INPUT:
            xA : [float] - współrzędna x punktu A
            yA : [float] - współrzędna y punktu A
            xB : [float] - współrzędna x punktu B
            yB : [float] - współrzędna y punktu B
        OUTPUT:
            (Az_deg, dist_AB) - krotka dwuelementowa, gdzie:
            Az_deg : [float] - azymut AB w stopniach dziesiętnych
            dist_AB: [float] - odległość AB w jednostkach jak podano współrzędne.
        EXAMPLE:    
            INP: xA =-45.00; yA = 23.82; xB = 67.98; yB = 34.12 
            RUN: az, dist = azimuth_dist_xy(xA, yA, x_B, y_b)
            OUT: 5.209060574544288, 113.44853635018832
            """
        # wyznaczenie przyrostów współrzednych
        dX = xB - xA
        dY = yB - yA 
        # wyznaczenie azymutu:
        if dX > 0 and dY > 0:                   # I ćwiartka (0-90st)
            Az      = atan(dY/dX)               # [rad]
            Az_deg  = degrees(Az)               # [deg]
        elif dX < 0 and dY > 0:                 # II ćwiartka (90-180st)
            Az      = atan(dY/dX)+  pi          # [rad]
            Az_deg  = degrees(Az)               # [deg]
        elif dX < 0 and dY < 0:                 # III ćwiartka (180-270st)
            Az      =  atan(dY/dX) +  pi        # [rad]
            Az_deg  =  degrees(Az)              # [deg]
        elif dX > 0 and dY > 0:                 # IV ćwiartka (270-360st)
            Az      =  atan(dY/dX)  + 2 *  pi   # [rad]
            Az_deg  =  degrees(Az)              # [deg]
        elif dX == 0 and dY > 0:                # (90st)
            Az      =  pi /2                    # [rad]
            Az_deg  =  degrees(Az)              # [deg]
        elif dX < 0 and dY == 0:                # (180st)
            Az      =  pi                       # [rad]
            Az_deg  =  degrees(Az)              # [deg]
        elif dX == 0 and dY < 0:                # (270st)
            Az      =  pi +  pi /2              # [rad]
            Az_deg  =  degrees(Az)              # [deg]
        elif dX > 0 and dY == 0:                # (360st lub 0st)
            Az1     = 0                         # [rad]
            Az_deg1 =  degrees(Az1)              # [deg]
            Az2     = 2*  pi                    # [rad]
            Az_deg2 =  degrees(Az2)              # [deg]
            Az_deg  = (Az_deg1, Az_deg2)
        # wyznaczenie długości odcinka AB
        dist_AB =  sqrt(dX**2 +dY**2) # meter
        return Az_deg, dist_AB
    
    def az_alf(self, fi, lam, H, fis, lams, Hs):
        """
        Funkcja obliczająca azymut i kąt elewacji.
        
        Parameters
        ----------
        fi: [float][stopnie dziesiętne] - szerokość geodezyjna
        lam: [float][stopnie dziesiętne] - długość geodezyjna
        H: [metry] - wysokość elipsoidalna
        fis: [float][stopnie dziesiętne] - szerokość geodezyjna drugiego pkt
        lams: [float][stopnie dziesiętne] - długość geodezyjna drugiego pkt
        Hs: [metry] - wysokość elipsoidalna drugiego pkt
        
        Returns
        -------
        azymut: [float][stopnie, minuty, sekundy] - azymut dwóch punktów
        zenitalny: [float][stopnie, minuty, sekundy] - kąt zenitalny dwóch punktów
        """        
        fi = radians(fi)
        fis = radians(fis)
        lam = radians(lam)
        lams = radians(lams)

        N = self.a/(sqrt(1 - self.ecc2*(sin(fi)**2)));
        Ns = self.a/(sqrt(1 - self.ecc2*(sin(fis)**2)));

        wRr = array([[(N+H)*cos(fi)*cos(lam)],
                     [(N+H)*cos(fi)*sin(lam)],
                     [(N*(1-self.ecc2)+H)*sin(fi)]]) #x,y,z 

        wRs = array([[(Ns+Hs)*cos(fis)*cos(lams)],
                     [(Ns+Hs)*cos(fis)*sin(lams)], 
                     [(Ns*(1-self.ecc2)+Hs)*sin(fis)]]) #x,y,z s
        R = wRs-wRr
        dlR = sqrt(R[0,0]**2 + R[1,0]**2 + R[2,0]**2)

        u = array([[cos(fi)*cos(lam)],
                   [cos(fi)*sin(lam)],
                   [sin(fi)]])
        n = array([[-sin(fi)*cos(lam)],
                   [-sin(fi)*sin(lam)],
                   [cos(fi)]])
        e = array([[-sin(lam)],
                   [cos(lam)],
                   [0]])
        wR = array([[R[0,0]/dlR],
                    [R[1,0]/dlR],
                    [R[2,0]/dlR]])

        wRt = transpose(wR)
        ut = transpose(u)
        alfa = atan((wRt @ e/(wRt @ n)))*180/pi+180;
        z = acos(ut @ wR)*180/pi

        azymut = geo.dms(alfa)
        zenitalny = geo.dms(z)
        return(azymut, zenitalny)
    
    def odl3D(x, y, z, x2, y2, z2):
        """
        Odległosć 3D
        
        Parameters:
        --------------
        x: [float] - współrzędna ortokartezjańska
        y: [float] - współrzędna ortokartezjańska
        z: [float] - współrzędna ortokartezjańska
        x2: [float] - współrzędna ortokartezjańska drugiego punktu
        y2: [float] - współrzędna ortokartezjańska drugiego punktu
        z2: [float] - współrzędna ortokartezjańska drugiego punktu
    
        Returns:
        --------------
        odl: [float][metry] - odległosć między punktami

        """
        A = [x, y, z]
        B = [x2, y2, z2]
        od = sqrt( (A[0] - B[0])**2 + (A[1] - B[1])**2 + (A[2] - B[2])**2 )
        return(od)
    
    
if __name__ == "__main__":
    # utworzenie obiektu
    geo = Transformacje(model = "wgs84")
    # dane XYZ geocentryczne
    #X = 3664940.500; Y = 1409153.590; Z = 5009571.170
    plik = "wsp_inp.txt"
    # odczyt z pliku: https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.genfromtxt.html
    tablica = genfromtxt(plik, delimiter=',', skip_header = 4)
    #print(tablica)
    X = tablica[:, 0]
    Y = tablica[:, 1]
    Z = tablica[:, 2]
    xyz = [tablica[:, 0],tablica[:, 1],tablica[:, 2]]
    savetxt("wsp_out.txt", tablica, delimiter=',', fmt = ['%10.2f', '%10.2f', '%10.3f'], header = 'Konwersja współrzednych geodezyjnych \\ Autor: Julia Szmul \n Dane do zadania: \n X \t\t Y \t Z')
    
    flh = zeros((12,10))
    for i,x in enumerate(array(xyz).T):

        flh[i,0], flh[i,1], flh[i,2] = geo.xyz2plh(x[0],x[1],x[2])
        flh[i,3], flh[i,4] = geo.u2000(flh[i,0], flh[i,1])
        flh[i,5], flh[i,6] = geo.u1992(flh[i,0], flh[i,1])
        flh[i,7], flh[i,8], flh[i,9] = geo.neu(x[0],x[1],x[2])
        
    raport = open("wsp_out.txt", 'a')
    raport.write(f' \n Transformacje: XYZ ->flh; flh->X2000,Y2000; flh->X1992,Y1992')
    raport.write(f' \n fi \t\t lambda \t h \n')
    for wiersz in flh:
        f = wiersz[0]
        l = wiersz[1]
        h = wiersz[2]
        raport.write(f' {f:.4f} \t {l:.4f} \t {h:.4f}\n')
        
    raport.write(f'\n X2000 \t\t\t Y2000 \n')
    for wiersz in flh:
        x0 = wiersz[3]
        y0 = wiersz[4]
        raport.write(f'{x0:.4f} \t\t {y0:.4f} \n')
    raport.write(f'\n X1992 \t\t\t Y1992 \n')
    for wiersz in flh:
        x9 = wiersz[5]
        y9 = wiersz[6]
        raport.write(f'{x9:.4f} \t\t {y9:.4f} \n')
    raport.close()
    print("\n Wyniki zapisano w pliku tekstowym!")