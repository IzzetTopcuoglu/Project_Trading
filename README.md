# day-trader

BTC-USDT (Binance Futures) uzerinde kisa vadeli (5 dakikalik mumlar)
al-sat kararlari uretmeyi hedefleyen, birden fazla modelin (fiyat
tahmini, haber sentiment'i, hacim analizi, orderbook analizi) bir
karar agacinda birlestigi bir sistemin ilk parcasi.

Bu ilk asamada sadece **fiyat tahmini yapan LSTM** modeli var. Diger
modeller (sentiment, hacim, orderbook) ve karar agaci ayni yapiya
zamanla eklenecek.

## Neden boyle tasarlandi

- **Zaman bazli train/val/test bolme, shuffle yok.** Zaman serisinde
  rastgele karistirma modelin gelecegi "gormesine" (leakage) yol acar.
- **Scaler sadece train verisine fit edilir.** Val/test istatistiklerinin
  sizmasi da bir leakage turudur.
- **Model fiyati degil, bir sonraki mumun log-getirisini tahmin eder**,
  sonra bu getiri guncel fiyata uygulanarak sayisal bir fiyat tahminine
  cevrilir (`train.py:return_to_price`). Ham fiyat serisi durağan
  olmadigi icin dogrudan fiyat tahmini genelde modelin "bir onceki
  fiyati tekrarlamasi" ile sonuclanir; getiri uzerinden gitmek daha
  saglikli bir sinyal verir.
- **Hicbir ozellik ham fiyat/hacim seviyesinde degil** (v2 guncellemesi).
  BTC aylar icinde ciddi trend yapabildigi icin (ornegin test setinde
  62k -> 82k), ham fiyati/hacmi dogrudan feature olarak kullanmak,
  train doneminde ogrenilen olceklendirmenin test doneminde geçerli
  olmamasina (covariate shift) yol acar. Bunun yerine `log_return`,
  `hl_range`, `oc_range`, `price_to_sma20`, `volatility_20`, `rsi_14`,
  `volume_z` gibi zamandan bagimsiz (stationary) ozellikler kullaniliyor.

## v1 sonucu ve v2 guncellemesi

Gercek Binance verisiyle ilk egitimde (5 dakikalik mumlar, ham OHLCV
feature'lari ile) `directional_accuracy` ~0.49 cikti — yani model
yon tahmininde yazi-tura atmaktan farksizdi. Bu beklenen bir sonuc:
kisa vadede sadece fiyat verisinden yon tahmini yapmak zaten zor, ve
yukarida bahsedilen covariate shift sorunu isi daha da zorlastiriyordu.

v2'de feature seti stationary hale getirildi (yukarida listelenen
7 ozellik). Bunu denedikten sonra hala ~0.50 civarinda cikarsa
sasirma — bu, "tek basina fiyat LSTM'i BTC yonunu guvenilir sekilde
tahmin edemiyor" sonucunu dogrular, ki zaten bu yuzden sentiment,
hacim ve orderbook modellerini de ekleyip bir karar agacinda
birlestirecegiz.

Gercekten de 5 dakikalik, 1 saatlik ve 4 saatlik mumlarla yapilan
denemelerin hepsinde `directional_accuracy` ~0.48-0.50 araliginda
cikti — yani zaman dilimini degistirmek tek basina bir edge
yaratmadi. Bu da bizi asagidaki v3 denemesine (makro/cross-asset
veri eklemek) yonlendirdi.

## v3 denemesi: makro/cross-asset veri

Fikir: BTC'nin kendi fiyat/hacim gecmisi disinda, genel piyasa
ortamini yansitan gunluk sinyaller eklemek modele biraz daha bilgi
verebilir mi? Eklenen ozellikler (`src/data/fetch_macro_data.py`
tarafindan cekilip `macro_daily` tablosuna yaziliyor):

- **Yaklasik toplam kripto piyasa degeri**: CoinGecko'dan en buyuk
  20 coin'in piyasa degerlerinin toplami (gercek "total market cap"
  degil, bir yaklastirma — CoinGecko'nun gercek `/global` gecmis
  verisi ucretsiz planda yok)
- **USDT toplam islem hacmi**: toplam kripto hacminin bir proxy'si
  olarak
- **S&P 500, altin, Brent petrol, gumus, ABD 10 yillik tahvil
  faizi**: yfinance uzerinden

Bunlar GUNLUK veri oldugu icin, 5 dakikalik BTC mumlarina eklerken
iki leakage riskine dikkat edildi (detay icin `dataset.py`'nin
basindaki docstring'e bak):

1. Makro degerler ham seviye olarak degil, gunluk log-getiri (faiz
   icin fark) olarak eklendi — BTC ozelliklerinde yaptigimiz
   stationary donusumun ayni mantigi.
2. Bir gunun BTC mumlari, O GUNUN DEGIL bir onceki gunun makro
   kapanis degisimini goruyor (tarih 1 gun ileri kaydirilarak) —
   cunku ornegin bugunku S&P 500 kapanisi, ABD borsasi kapanana
   kadar gercekte bilinmiyor.

### Nasil denenir

1. [coingecko.com/en/api/pricing](https://www.coingecko.com/en/api/pricing)
   uzerinden ucretsiz bir "Demo" API key al (kredi karti gerekmez).
2. `.env` dosyana ekle: `COINGECKO_API_KEY=senin-key-in`
3. Makro veriyi bir kere cek (gunluk veri oldugu icin siklikla
   tekrar cekmen gerekmez):
   ```bash
   python -m src.data.fetch_macro_data
   ```
4. `.env` dosyasinda `INCLUDE_MACRO=true` yap.
5. Modeli yeniden egit:
   ```bash
   python -m src.models.lstm_price.train
   ```
6. Ciktidaki `directional_accuracy`'yi, `INCLUDE_MACRO=false` ile
   aldigin ~0.48-0.50 sonucuyla karsilastir. Belirgin bir iyilesme
   olursa (ornegin ~0.52-0.53+) makro veri kalici olarak feature
   setine eklenebilir; yoksa bu da yine "tek basina yeterli degil,
   sentiment/hacim/orderbook modelleri ile devam" sonucunu
   dogrulayan bir bulgu olur.

Not: `INCLUDE_MACRO=true` iken model dosyasi 14 ozellikle
egitildigi icin, daha sonra `INCLUDE_MACRO=false` ile tahmin
yapmaya calisirsan (veya tam tersi) `predict.py` scaler/model
boyut uyusmazligindan hata verir — ikisini karsilastirirken her
ayarda hem `train.py` hem `predict.py`'yi ayni `INCLUDE_MACRO`
degeriyle calistirdigindan emin ol (`config.py` `.env`'den okudugu
icin bu otomatik tutarli olur, sadece iki ayri denemeyi ayni anda
karistirma).

## Farkli zaman dilimlerini denemek

Kod herhangi bir mum araligiyla calisacak sekilde yazildi, yani 1
saatlik veya 4 saatlik mumları denemek icin TEK YAPMAN GEREKEN
`.env` dosyasini degistirip veriyi/modeli yeniden calistirmak:

```
TIMEFRAME=1h
LOOKBACK_DAYS=365
```

sonra:

```bash
python -m src.data.fetch_data
python -m src.models.lstm_price.train
```

Daha uzun zaman dilimleri (1s, 4s) piyasa gurultusunu azaltir, bu da
genelde 5 dakikalik mumlara gore daha yuksek bir directional_accuracy
ihtimali demek — ama ayni zamanda daha az sayida mum (daha az egitim
verisi) anlamina da geliyor, bu yuzden `LOOKBACK_DAYS`'i arttirmak
gerekebilir (1s icin 365 gun ~8760 mum, 4s icin 365 gun ~2190 mum verir).

## Kurulum

```bash
git init
python -m venv .venv
# Windows:
.venv\Scripts\activate
# (mac/linux icin: source .venv/bin/activate)

pip install -r requirements.txt

copy .env.example .env    # Windows
# cp .env.example .env    # mac/linux
```

`.env` dosyasini acip istersen `SYMBOL`, `TIMEFRAME`, `LOOKBACK_DAYS`,
`WINDOW` degerlerini degistirebilirsin. Binance API anahtari public
OHLCV verisi cekmek icin GEREKLI DEGIL, bos birakabilirsin.
`COINGECKO_API_KEY` ve `INCLUDE_MACRO` sadece asagidaki "v3 denemesi:
makro/cross-asset veri" bolumunu denerken lazim, varsayilan olarak
bos/`false` birakabilirsin.

## v4 denemesi: yon tahminini dogrudan siniflandirma olarak yapmak

v1-v3'te hep ayni yontemi kullandik: modeli SAYISAL bir deger (log-getiri)
tahmin edecek sekilde egitip (regresyon, MSE loss), sonra bu tahminin
isaretine (+/-) bakarak yon cikariyorduk. Ama asil onemsedigimiz sey hep
yondu — MSE'yi optimize etmek, yonu dogru tahmin etmekle ayni sey degil.

`src/models/lstm_direction/` altinda, `lstm_price`'a DOKUNMADAN (o
deneyler/sonuclar hala gecerli), yeni bir siniflandirma modeli eklendi:

- Ayni 7 (+opsiyonel 7 makro) stationary ozellik kullanilir
- Hedef: bir sonraki mumun getirisi pozitif mi (1/"yukari") negatif mi
  (0/"asagi") — sayisal deger degil, ikili etiket
- Model, sigmoid cikis katmani + binary cross-entropy loss ile egitilir
  (`build_lstm_classifier`) - yani DOGRUDAN yon dogrulugunu optimize eder

### Neden sonucu sadece `accuracy`'ye degil, `majority_class_baseline`'a da bakarak degerlendirmeliyiz

Eger test setinde mesela mumlarin %53'u "yukari" kapaniyorsa, model hicbir
sey ogrenmeden surekli "yukari" tahmin ederek bile ~%53 dogruluk
gosterebilir - bu sahte bir basari olur. Bu yuzden `train.py` her zaman
iki seyi yan yana raporlar:

- `accuracy`: modelin gercek test dogrulugu
- `majority_class_baseline`: test setindeki en sik gorulen yonu surekli
  tahmin etseydik alacagimiz dogruluk (yani "hicbir sey ogrenmeden"
  ulasilabilecek taban)

Modelin gercekten bir sinyal yakaladigini soyleyebilmemiz icin
`accuracy`'nin `majority_class_baseline`'in ACIKCA UZERINDE olmasi
lazim - `train.py` bu farki (`edge`) da hesaplayip yazdiriyor, negatifse
uyari veriyor.

### Nasil denenir

```bash
python -m src.models.lstm_direction.train
python -m src.models.lstm_direction.predict
```

`INCLUDE_MACRO` ayari burada da gecerli (ayni `.env` degiskeni).

## v5 denemesi: XAUT (7/24 islem goren altin) ile makro-altin verisini degistirmek

v3'teki makro veri (`gold`, yfinance uzerinden) sadece ABD borsa
saatlerinde guncelleniyor - hafta sonu/tatil gunlerinde veri yok, bu
yuzden `dataset.py` bu gunleri ffill ile dolduruyordu (bkz. yukarida).

**XAUT** (Tether Gold), Binance'de BTC gibi 7/24 islem goren, altina
bagli bir kripto token. Akademik olcumlere gore spot altin fiyatini cok
yakin takip ediyor (ortalama sapma ~0.03 $/ons). Bunun onemli bir
avantaji var: XAUT, BTC ile TAM AYNI mum araliginda (5dk/1s/vb.) ve ayni
anda kapaniyor - yani makro veri gibi GUNLUK degil, BTC'nin kendi
ozellikleri (log_return, rsi_14, vb.) gibi ayni cozunurlukte. Bu
yuzden:

- 1 gun ileri kaydirma (leakage onlemi) GEREKMEZ - bir mumun
  `xaut_log_return`'u, o mumun kendi `log_return`'u ile tam ayni anda
  biliniyor (`add_xaut_feature`, `lstm_price/dataset.py`)
- Hafta sonu ffill'e de gerek yok - XAUT hic durmadan islem goruyor

### Nasil denenir

```bash
python -m src.data.fetch_xaut_data
```

Sonra `.env`'de `INCLUDE_XAUT=true` yap ve istedigin modeli (`lstm_price`
veya `lstm_direction`) tekrar egit:

```bash
python -m src.models.lstm_price.train
# veya
python -m src.models.lstm_direction.train
```

`INCLUDE_MACRO` ve `INCLUDE_XAUT` birbirinden bagimsiz, ikisini ayni
anda da acabilirsin (o zaman hem gunluk makro hem XAUT ozellikleri
eklenir - dikkat: bu durumda hem `macro_gold_ret` hem `xaut_log_return`
ayni anda modele girer, ikisi de altini temsil ettigi icin biraz
fazlalik/redundan olabilir - istersen sadece `INCLUDE_XAUT=true`
birakip `INCLUDE_MACRO`'yu diger varliklar (S&P500, gumus, petrol,
tahvil) icin kullanmaya devam edebilirsin).

## v6 denemesi: orderbook (emir defteri) entegrasyonu

Bitirme projesi kapsaminda planlanan ucuncu sinyal kaynagi: Binance
Futures'in anlik emir defterinden (bid/ask derinligi) alis/satis baskisi
dengesizligini (order book imbalance) hesaplamak.

**Onemli bir kisit var ve bu, tasarimi dogrudan etkiliyor:** Binance'in
(ve genel olarak hicbir borsanin ücretsiz) REST API'si GECMISE DONUK
orderbook vermiyor - sadece O ANKI emir defterini gorebiliyorsun. Yani
XAUT/makro veride yaptigimiz gibi "son 180 gunu cek, LSTM'in egitim
verisine ozellik olarak ekle" burada mumkun degil; kimse gecmisteki her
anin emir defterini ucretsiz saklamiyor.

Bu yuzden orderbook, mimaride LSTM'in (Katman 1) degil, **LLM Agent'in
(Katman 2) canli karar anina ait bir sinyali** olarak kurgulandi (bkz.
proje dokumanindaki akis semasi):

- `src/data/orderbook.py` -> `get_orderbook_context()`: o anki
  emir defterini ceker, alis/satis hacmini ve dengesizligi
  (`imbalance`, -1..+1 arasi) hesaplar. LLM Agent yazildiginda bu
  fonksiyon dogrudan onun girdisine baglanacak.
- `src/data/log_orderbook.py`: tek bir anlik goruntuyu DuckDB'de
  `orderbook_snapshots` tablosuna EKLER (fetch_data.py gibi tabloyu
  sifirlamaz, biriktirir). Bunu Windows Gorev Zamanlayicisi ile
  `TIMEFRAME` ile ayni sikilikta (varsayilan 5 dakikada bir) tekrar
  tekrar calistirirsan, proje suresince kendi gecmis orderbook
  veritabanini biriktirmis olursun - yeterli veri birikirse ileride
  LSTM'e denemelik bir ozellik olarak eklenebilir, ama bu projenin
  suresi icinde garanti degil.

### Nasil denenir

```bash
# Tek seferlik anlik ozet (agent'in gorecegi seyin aynisi)
python -m src.data.orderbook

# DuckDB'ye tek bir satir ekler - duzenli araliklarla calistirilirsa gecmis birikir
python -m src.data.log_orderbook
```

Sentetik veriyle (agsiz) dogrulama testleri icin proje kokunde:

```bash
python tests_orderbook.py
```

## v7 denemesi: ekonomik takvim (FOMC/CPI/NFP) bayragi

`src/data/economic_calendar.py`, Federal Reserve (FOMC) ve BLS (CPI,
NFP/Employment Situation) resmi takvimlerinden elle derlenmis STATIK
bir liste tutuyor (2025 sonu - 2026 sonu arasi). Orderbook'un aksine
bu GECMISE de uygulanabilir - bu tarihler aylar oncesinden kamuya acik
oldugu icin leakage riski yok (bkz. dosyanin basindaki not).

Ekler:
- `calendar_days_to_next_event`: bir sonraki yuksek etkili olaya kac gun kaldigi
- `calendar_high_impact_soon`: bu olay `window_days` (varsayilan 1) gun icinde mi (0/1)

### Nasil denenir

```bash
# Tek seferlik: bir sonraki olayi gosterir (agent'in gorecegi seyin aynisi)
python -m src.data.economic_calendar
```

Sonra `.env`'de `INCLUDE_CALENDAR=true` yap ve istedigin modeli tekrar egit
(`INCLUDE_MACRO`/`INCLUDE_XAUT` gibi bagimsiz calisir, istersen hepsini
ayni anda acabilirsin):

```bash
python -m src.models.lstm_direction.train
```

Testler (agsiz, sentetik veriyle - `prepare_dataset`/`prepare_direction_dataset`
ile tam entegrasyonu da dogrular):

```bash
python tests_calendar.py
```

**Onemli bakim notu:** liste 2026 sonuna kadar gidiyor. Yeni yil
oldugunda ya da listenin sonuna yaklasildiginda, `economic_calendar.py`
icindeki `_FOMC_DECISION_DAYS` / `_CPI_RELEASE_DAYS` / `_NFP_RELEASE_DAYS`
listelerini dosyanin basindaki resmi kaynaklardan guncellemen gerekiyor.

## v8: LLM Agent (Katman 2)

`src/agent/` paketi, Katman 1'in (LSTM) yon tahminini, orderbook
dengesizligini ve ekonomik takvim riskini birlestirip yapilandirilmis
bir alim-satim karari ureten LLM Agent'i icerir:

- `prompts.py`: versiyonlanmis (`PROMPT_VERSION`) sistem promptu - karar
  kuralini (sinyaller uyumluysa 5x, degilse 1x, cok belirsizse no_trade)
  acikca tanimliyor, boylece model kendi bagimsiz kuralini uydurmuyor.
- `schema.py`: kararin uymasi GEREKEN JSON semasi (`direction`,
  `confidence`, `leverage` [1 ya da 5], `stop_limit_pct`,
  `signals_aligned`, `rationale`). Bu sema Anthropic API'sine bir
  "tool" olarak veriliyor ve model bu tool'u cagirmaya ZORLANIYOR
  (`tool_choice`) - yani serbest metin degil, semaya uyan bir JSON
  garanti ediliyor. Ayrica `jsonschema` ile bir kez daha dogrulaniyor.
- `llm_agent.py`: `decide(context)` fonksiyonu. **Herhangi bir hatada**
  (API anahtari yok, ag hatasi, semaya uymayan cikti, timeout) ASLA
  exception firlatmiyor - `SAFE_DEFAULT_DECISION`'a (direction=no_trade)
  duser. Agent otonom calistigi ve karari dogrudan Katman 3'e (gercek
  emir) gittigi icin, "bilmiyorum" durumunda islem acmamak tek guvenli
  secenek.

### Kurulum ve deneme

`.env`'de `ANTHROPIC_API_KEY` ve istersen `LLM_MODEL`'i doldur (guncel
model id'leri icin https://docs.claude.com/en/docs/about-claude/models).
Sonra LSTM (`lstm_direction`) egitilmis olmali, gercek Binance
baglantisiyla canli bir karar denemek icin:

```bash
python -m src.agent.llm_agent
```

Testler (GERCEK API'ye baglanmaz, hepsi mock):

```bash
python tests_agent.py
```

## v9: Otomasyon (Katman 3 - emir gonderme + Telegram)

`src/automation/` paketi:

- `risk_state.py`: `data/risk_state.json` icinde basit bir durum
  tutar - gunluk oturum basi sermaye, ve TEK bir `stopped` bayragi.
  Bu bayrak iki kaynaktan gelebilir: Telegram'dan manuel `/durdur`
  komutu, ya da `check_global_loss_limit()`'in otomatik tetiklemesi
  (gunluk oturum basi sermayeye gore `GLOBAL_LOSS_LIMIT_PCT`'i asan
  kayip). Hangisi olursa olsun, `place_trade()` yeni islem acmaz.
- `order_execution.py`: `place_trade(decision)` - LLM Agent'in
  kararini ccxt ile Binance Futures'a (piyasa emriyle giris + stop-limit
  koruma) gonderir. **`DRY_RUN=true` (varsayilan) iken gercek emir
  GONDERILMEZ**, sadece plan loglanir/donulur - gercek parayla
  denemeden once mutlaka DRY_RUN=true ile birkac tur calistirip
  ciktilari incele.
- `telegram_bot.py`: `send_status_message()` her karari/sonucu
  Telegram'a bildirir; `/durdur`, `/devam`, `/durum` komutlarini
  isler. **Onay mekanizmasi DEGIL** - agent kendi basina islem acar,
  Telegram sadece izleme + acil durdurma icin.

### Kurulum

1. Telegram'da @BotFather'a `/newbot` yaz, verdigi token'i
   `TELEGRAM_BOT_TOKEN`'a koy.
2. Botuna herhangi bir mesaj at, sonra tarayicida
   `https://api.telegram.org/bot<TOKEN>/getUpdates` adresini ac -
   donen JSON'daki `message.chat.id` senin `TELEGRAM_CHAT_ID`'n.
3. `.env`'de ikisini de doldur.

### Nasil denenir

```bash
# Telegram komutlarini bagimsiz dinlemek icin (Ctrl+C ile durdur)
python -m src.automation.telegram_bot
# Baska bir terminalde/telefonundan botuna /durum, /durdur, /devam yaz
```

Testler (GERCEK Binance/Telegram API'sine baglanmaz, hepsi mock):

```bash
python tests_automation.py
```

### Hepsini birlikte calistirmak

```bash
python -m src.main
```

Bu, Katman 1 -> 2 -> 3'u tek bir dongude birlestirir (`TIMEFRAME`
araligiyla tekrarlar). Once mutlaka `DRY_RUN=true` ile birkac tur
calistir, konsoldaki ve Telegram'daki ciktilari incele.

## v10: Binance Futures Testnet ile uctan uca dogrulama

`DRY_RUN=true` sadece "ne yapardim" planini loglar, gercekten bir
borsa API'sine emir GONDERMEZ. Uctan uca akisi (kaldirac ayarlama,
emir olusturma, stop-limit emri) gercek bir borsa API'sine karsi ama
SIFIR risk ile dogrulamak icin Binance Futures **Testnet**'ini
(sahte parayla calisan, ayri bir Binance ortami) kullanabilirsin.

**Onemli tasarim tercihi:** fiyat/orderbook verisi (LSTM'in ve
LLM Agent'in gordugu sinyaller) HER ZAMAN gercek Binance'ten gelmeye
devam ediyor - sadece emir gonderme/bakiye kismi testnet'e yonleniyor.
Testnet'teki emir defteri gercekci degil (az katilimci, ince likidite),
o yuzden sinyal olarak kullanilmiyor; ama emir gonderme KODUNU
(`order_execution.py`) gercek Binance Futures API'sine karsi test
etmiş oluyoruz.

### Kurulum

1. https://testnet.binancefuture.com adresine git, GitHub hesabinla
   giris yap (mainnet hesabından TAMAMEN AYRI bir ortam).
2. Testnet hesabına otomatik olarak sahte USDT tanimlanir (yoksa
   sayfadaki "Faucet" ile isteyebilirsin).
3. API Key olustur (yine testnet sayfasi uzerinden), `.env`'e yaz:
   ```
   BINANCE_TESTNET=true
   TESTNET_API_KEY=...
   TESTNET_API_SECRET=...
   ```
4. `DRY_RUN=false` yap - **ama BINANCE_TESTNET=true oldugu surece
   gercek paran risk altinda degil**, emirler sadece testnet'e gider.

### Nasil denenir

```bash
python -m src.main
```

Konsolda gercek bir `entry_order`/`stop_order` ciktisi gormen lazim
(artik "DRY_RUN aktif" mesaji yerine). Testnet hesabini
testnet.binancefuture.com uzerinden acip pozisyonun gercekten acildigini
da gorebilirsin. Testler (mock ile, agsiz):

```bash
python tests_automation.py
```

**Sirali guvenlik kontrolu (ozet):**

| Durum | Sonuc |
|---|---|
| `DRY_RUN=true` | Hicbir yere emir gitmez, sadece log |
| `DRY_RUN=false` + `BINANCE_TESTNET=true` | Emir gider ama SAHTE paraya (testnet) |
| `DRY_RUN=false` + `BINANCE_TESTNET=false` | Emir GERCEK hesabina, GERCEK parayla gider |

Gercek paraya sadece testnet'te birkac tur sorunsuz calistiktan, ve
`order_execution.py`'deki stop-limit emrinin (STOP_MARKET + reduceOnly)
testnet'te dogru calistigini konsol/testnet arayuzunden gordukten
sonra gec.

**Durum: dogrulandi (2026-09-20).** `run_testnet_order_test.py` ile
testnet'e karsi gercek bir market giris emri + STOP_MARKET/reduceOnly
korumali emir basariyla gonderildi (`status: executed`, gercek
`entry_order`/`stop_order` id'leriyle). Not: ccxt 4.x, binanceusdm'de
`set_sandbox_mode(True)` sonrasi kendiliginden bir `NotSupported`
hatasi firlatiyor ("futures testnet artik resmi desteklenmiyor" diyerek)
- ama testnet.binancefuture.com fiilen calisiyor, bu yuzden
`_make_exchange()` icinde `exchange.options["disableFuturesSandboxWarning"] = True`
ile bu uyari bilerek bastiriliyor. Ayrica: protective stop emri
Binance'de klasik bir "order" degil, bir "algo order" (`algoId`) olarak
donuyor - ccxt bunu unified `create_order` cagrisi uzerinden dogru
sekilde yonetiyor, ek bir kod degisikligi gerekmedi.

## Onemli: Binance erisimi

Binance, bazi bulut/VPS IP araliklarini engelliyor. Bu betikleri
kendi bilgisayarindan (Turkiye'den) calistirmalisin; VPS'ten veya bir
bulut ortamindan calistirirsan `fetch_data.py` muhtemelen
`NetworkError`/geo-block hatasi alacaktir.

## Calistirma sirasi

```bash
# 1) Tarihsel 5dk'lik BTC/USDT futures verisini indirip DuckDB'ye yazar
python -m src.data.fetch_data

# 1b) (opsiyonel, bkz. "v3 denemesi") makro/cross-asset gunluk veriyi ceker
python -m src.data.fetch_macro_data

# 2) LSTM'i egitir, artifacts/ altina modeli + scaler'i + grafikleri kaydeder
python -m src.models.lstm_price.train

# 3) En guncel veriyle bir sonraki mumun fiyatini tahmin eder
python -m src.models.lstm_price.predict
```

`train.py` calistiginda konsolda su metrikleri gorursun:

- `mae_log_return` / `rmse_log_return`: getiri olceginde hata
- `directional_accuracy`: modelin yon (yukselis/dusus) tahmininin
  ne kadar dogru oldugu — **rastgele bir modelde bile ~%50 civarinda
  cikar**, bunun anlamli olup olmadigini zamanla degerlendirecegiz
- `mae_price_usdt`: fiyat olceginde ortalama mutlak hata

`artifacts/lstm_price_test_predictions.png` grafiginde gercek fiyat
ile tahmin edilen fiyati gorsel olarak karsilastirabilirsin.

## Klasor yapisi

```
day-trader/
  .env.example        # ornek ortam degiskenleri
  requirements.txt
  src/
    config.py          # .env okuma, ortak ayarlar
    data/
      fetch_data.py       # Binance Futures'tan BTC OHLCV cekip DuckDB'ye yazar
      fetch_macro_data.py # CoinGecko + yfinance'dan gunluk makro veri cekip DuckDB'ye yazar
      fetch_xaut_data.py  # Binance Futures'tan XAUT (7/24 altin) OHLCV cekip DuckDB'ye yazar
      orderbook.py         # anlik emir defteri + alis/satis dengesizligi (LLM Agent icin canli sinyal)
      log_orderbook.py     # tek bir orderbook anlik goruntusunu DuckDB'ye ekler (gecmis biriktirmek icin)
      economic_calendar.py # FOMC/CPI/NFP statik takvimi + gecmise/canliya ozellik uretimi
    models/
      lstm_price/
        dataset.py       # ozellik muhendisligi, zaman-bazli split, pencereleme
        model.py          # Keras LSTM mimarisi (regresyon)
        train.py          # egitim + degerlendirme + kaydetme
        predict.py        # en guncel veriyle tek adim ileri fiyat tahmini
      lstm_direction/
        dataset.py       # lstm_price'daki ozellik muhendisligini kullanir, hedefi ikili yon etiketine cevirir
        model.py          # Keras LSTM mimarisi (siniflandirma: sigmoid + binary cross-entropy)
        train.py          # egitim + degerlendirme (majority-class baseline karsilastirmasiyla) + kaydetme
        predict.py        # en guncel veriyle tek adim ileri yon (P(yukari)) tahmini
    agent/                # Katman 2 - LLM Agent
      prompts.py           # versiyonlanmis sistem promptu
      schema.py             # zorunlu JSON karar semasi (Anthropic tool use)
      llm_agent.py          # decide() - hata durumunda guvenli varsayilana duser
    automation/           # Katman 3 - otomasyon
      risk_state.py         # /durdur + global zarar kurali icin ortak "stopped" durumu
      order_execution.py    # ccxt ile Binance'e emir gonderme (DRY_RUN korumali)
      telegram_bot.py       # durum bildirimi + /durdur, /devam, /durum komutlari
    main.py               # Katman 1->2->3'u birlestiren ana dongu
  data/                 # (gitignore'da) DuckDB veritabani, risk_state.json burada olusur
  artifacts/            # (gitignore'da) egitilmis model, scaler, grafikler
```

## Git'e pushlama

Her asamadan sonra geri donebilmen icin kucuk commit'ler oneririm,
ornegin:

```bash
git add .
git commit -m "lstm fiyat modeli: veri pipeline + egitim + tahmin"
git remote add origin <senin-repo-linkin>
git push -u origin main
```

`.gitignore` `.env`, `data/` ve `artifacts/` klasorlerini disarida
birakiyor — API anahtarlarin ve buyuk model/veri dosyalarin git'e
girmeyecek.

## Sirada ne var (bitirme projesi kapsaminda)

Uc katman da (kod olarak) tamamlandi - sirada asagidaki gibi ADIM ADIM
kendi ortaminda deneyip dogrulamak var (her adimin testleri agsiz/mock,
ama gercek entegrasyon senin Binance/Anthropic/Telegram hesaplarinla
calisir, o yuzden burada uctan uca test edilmedi):

1. ~~Orderbook analizi (canli sinyal)~~ - v6, `src/data/orderbook.py`
2. ~~Ekonomik takvim bayragi~~ - v7, `src/data/economic_calendar.py`
3. ~~LLM Agent~~ - v8, `src/agent/` (Anthropic API key gerekiyor)
4. ~~Otomasyon (emir gonderme + Telegram)~~ - v9, `src/automation/`
   (DRY_RUN=true ile once dene!)
5. ~~Hepsini `python -m src.main` ile DRY_RUN=true olarak izlemek~~
6. ~~Binance Futures Testnet ile uctan uca, sahte parayla dogrulama~~ -
   v10, `BINANCE_TESTNET=true` + `DRY_RUN=false` - basariyla dogrulandi
   (2026-09-20): gercek market giris + STOP_MARKET/reduceOnly emri
   testnet'te calisti (bkz. `run_testnet_order_test.py`)
7. Testnet'te birkac tur DAHA (farkli yon/leverage kombinasyonlariyla)
   sorunsuz calistiktan sonra, bilerek ve kucuk miktarla,
   `BINANCE_TESTNET=false` yapip gercek (ama kucuk) bir islemi dogrula
8. (Parkta, sonra donulecek) LSTM'in directional_accuracy'sini %50'nin
   anlamli sekilde uzerine cikarma calismasi
