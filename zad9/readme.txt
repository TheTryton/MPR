1. wyniki pomiarów wraz z wykresami zamieszczam w folderze dane (wykresy wygenerowane za pomocą main.py)
2. zmodyfikifowany kod zamieszczam w folderze kod (zmodyfikowałem kod tak aby mierzył czasy dla rozmiarów od 2^16 do 2^28)
   + dodałem wsparcie CMake dla prostszej kompilacji podprojektów
3. Wnioski: najlepsze czasy redukcji dały algorytmy atomic block i atomic warp (atomic simple był wolniejsze najprawdopodobniej ze względu
   "memory congestion" na 1 elemencie tablicy wyjściowej) dodatkowo dobrze się zachowywały jeżeli chodzi o błąd dodawania zmiennoprzecinkowego
   co w przypadku pozostałych algorytmó dla dużej liczby elementów tablicy powodowało rozjazd wartości końcowej między urządzeniem a hostem.
   Co do generacji histogramów - algorytm naive działał szybciej dla małej liczby kubełków natomiast wraz ze wzrostem liczby kubełków
   algorytm simple zaczynał powoli doganiać algorytm naive a w przypadku liczby kubełków 1000 był szybszy