# Poređenje reprezentacija slike za klasifikaciju saobraćajnih znakova pri kontrolisanim degradacijama ulaza

Projekat iz predmeta **Računarski vid**, Elektronski fakultet u Nišu.
Mihajlo Madić, 2119.

## Sažetak

Pri izboru reprezentacije slike za klasifikaciju saobraćajnih znakova uobičajeno je da odluku
donese tačnost na čistim podacima, a ovaj rad ispituje da li je takav kriterijum dovoljan.
Upoređene su četiri reprezentacije nad GTSRB skupom podataka, uz namerno strogu kontrolu. To
su PCA, HOG, metod vreća vizuelnih reči nad gustim SIFT deskriptorima i konvoluciona neuronska
mreža u dve varijante. Klasifikator, pretprocesiranje i protokol evaluacije drže se fiksnim,
tako da razlike u rezultatima ostaju pripisive isključivo reprezentaciji.

Umesto jednog broja po metodi, svaka konfiguracija je izmerena i na čistom test skupu i pod
tri kontrolisane degradacije, svaka na pet nivoa jačine. Degradacije su šum, zamućenje usled
kretanja i gama korekcija.

Glavni uvid je da tačnost na čistim podacima ne naznačava nužno **otpornost** na određenu
vrstu degradacije. Pod jakim šumom redosled uspešnosti se preokreće, metoda koja je na čistom
ulazu poslednja postaje najotpornija i nadmašuje konvolucionu mrežu na istim pikselima, dok
metoda koja se oslanja na gradijente gubi gotovo sve. Pod zamućenjem i gama korekcijom tog
preokreta nema, pa je interakcija između reprezentacije i degradacije stvarna, ali vezana za
*vrstu* degradacije.

Pre početka eksperimenata, napravljen je skup hipoteza o ponašanju svake reprezentacije, pri
određenoj vrsti šuma. Većina ovih pretpostavki se obistinila, a dva promašaja su analizirana,
jer su informativnija od onih koja su se potvrdila. Praktična posledica je da se
reprezentacija bira prema degradaciji koja u datoj primeni dominira, jer rang-lista na čistim
podacima za tu odluku nije dovoljna.

**Ključne reči:** reprezentacije slike, GTSRB, PCA, HOG, BoVW, CNN, klasifikacija.

## Dokumenti

- [Završni izveštaj](docs/report/main.pdf)
- [Predlog projekta](docs/proposition/predlog_projekta_compvis_mihajlo_madic_2119.pdf)

---

Detaljan opis metodologije, rezultata i strukture repozitorijuma nalazi se u
[DETAILED_INFO.md](DETAILED_INFO.md).
