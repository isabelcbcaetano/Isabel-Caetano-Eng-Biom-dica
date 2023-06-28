

syms A0 k1 k2 beta gamma alpha A0 t

Ap = beta * k1 * A0/(k2-k1) + gamma*A0;
Ah = -1*beta*k1*A0/(k2-k1);

%hepatic activity
aht = Ap*exp(-k1*t) + Ah*exp(-k2*t);

clc;
disp(''); disp('');
disp('hepatic activity (ah)');
pretty(aht);

%blood activity
apt = A0*alpha*exp(-k1*t);

disp(''); disp('');
disp('blood activity (ap)');
pretty(apt);


%Laplace transform
Lap = laplace(apt);
Lah = laplace(aht);

disp(''); disp('');
disp('Laplace transform of the hepatic activity (Lah)');
pretty(Lah);

disp(''); disp('');
disp('Laplace transform of the blood activity (Lap)');
pretty(Lap);

%laplace transform of the retention curve
Hs = Lah/Lap;

disp(''); disp('');
disp('Laplace transform of the retention curve (hs)');
pretty((Hs));

%invert the Laplace transform
ht = ilaplace(Hs);

disp(''); disp('');
disp('Retention curve (ht)');
pretty((ht));




% uma tentativaaaa
Ah_conv_s=Lap*Hs;
ah_conv_t=ilaplace(Ah_conv_s);

disp(''); disp('');
disp('ah curve conv (ap_conv)');
pretty((ah_conv_t));

pretty((ah_conv_t)*t);

integral_num=int((ah_conv_t)*t,t);
disp(integral_num)

integral_den=int((ah_conv_t),t);
disp(integral_den)
