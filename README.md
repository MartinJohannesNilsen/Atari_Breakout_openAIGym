# DQN på Atari Breakout

En implementasjon av Deep Q-Network på retrospillet Atari Breakout. Det ble skrevet en rapport i forbindelse med prosjektet, som er tilgjengelig [her](https://firebasestorage.googleapis.com/v0/b/portfoliobymartinnilsen.appspot.com/o/Files%2FMaskinlæringsrapport_DQN_AtariBreakout.pdf?alt=media&token=114e9a0e-16bd-4710-b295-1c81b5edd809). For å teste agenten ble miljøet `BreakoutDeterministic-v4` fra OpenAI Gym tatt utgangspunkt i, og også senere modifisert. Koden er skrevet i Python, med pakker som bla. PyTorch og OpenCV.

----
## Spill spillet selv

Vedlagt i kildekoden ligger det en fil `breakout_test.py` som gir deg muligheten til å spille spillet selv. Dette er fra et av mine egne forsøk:

![24_meg](https://github.com/Martinnilsen99/Atari_Breakout_openAIGym/blob/master/ReadMe/Gifs/24_meg.gif)

Jeg klarte kun oppnå en score på 24, det var ikke så lett som det kan se ut som!

---
## Kjøre koden

Dette prosjektet tar i bruk pakken `argh`, som gjør det mulig å definere ulike parametere i terminalkommandoen for kjøring.
```
Grunnleggende:
$ python agent.py

Utvidelser:
-t               : Kjør i testmodus med en rendret versjon av miljøet. Trener ikke, og bør tas utgangspunkt i en trent modell.
-l               : Kjør med lokal synkronisering mot wandb.ai
-n <"navn">      : Definer navnet på kjøringen (vil bli spurt om dersom ikke testmodus)
-c <"model.pth"> : Sett en tidligere modell i /Models som utgangspunkt
```

En kan for eksempel kjøre `$ python agent.py -t -c "rmsprop_før_379.pth"` for å kjøre en modell som klarer oppnå en poengsum på 360. Ettersom systemet er deterministisk skal du få det samme ved kjøring.

---
## De tre beste kjøringene

### \# 1

![381_increasedMaxLenFrom_368_30m_last_targ_model.gif](https://github.com/Martinnilsen99/Atari_Breakout_openAIGym/blob/master/ReadMe/Gifs/381_increasedMaxLenFrom_368_30m_last_targ_model.gif "Boltzmann med justert handlingsrom og økt makslengde på handlinger i epoke. Score på 381.")

Boltzmann med redusert handlingsrom. Score på 381 etter 2t trening med utgangspunkt i modellen til \# 3.

### \# 2

![379_36m_RMSProp.gif](https://github.com/Martinnilsen99/Atari_Breakout_openAIGym/blob/master/ReadMe/Gifs/379_36m_RMSProp.gif "Boltzmann med RMSProp. 379 etter 36m steg.")

Boltzmann med RMSProp oppnådde en score på 379 etter 36m steg.

### \# 3

![368_30m_boltzreduced.gif](https://github.com/Martinnilsen99/Atari_Breakout_openAIGym/blob/master/ReadMe/Gifs/368_30m_boltzreduced.gif "Boltzmann med justert handlingsrom. 368 etter 30m steg.")

Boltzmann med redusert handlingsrom. Score på 368 etter 30m steg, og mister ikke siste livet. Ble avsluttet etter den nådde maksgrensa for antall handlinger i en episode; en begrensning satt i det originale miljøet for å unngå at den kjører uendelig dersom agenten bla. ikke starter spillet selv.

## En gang det ikke gikk like bra

![0_22m_boltzeps.gif](https://github.com/Martinnilsen99/Atari_Breakout_openAIGym/blob/master/ReadMe/Gifs/0_22m_boltzeps.gif "Boltzmann med synkende epsilon. Uendelig løkke ettersom den ikke starter spillet.")

Et eksempel på nevnt scenario over. Agenten starter ikke spillet, men vibrerer bare ved å gå hurtig fra høyre til venstre.

## Til sammenlikning: Første kjøring

![36_15m_v0.gif](https://github.com/Martinnilsen99/Atari_Breakout_openAIGym/blob/master/ReadMe/Gifs/36_15m_v0.gif "Første fungerende kjøring i Breakout-v0. Score på 36 etter 15m steg.")

Første fungerende kjøring i miljløet *Breakout-v0*. Oppnådd score på 36 etter 15m steg. Som beskrevet i rapporten ble miljøet *BreakoutDeterministic-v4* brukt i videre kjøringer. Dette fordi en ønsket ha mer kontroll over tilfeldighetene, mer om dette i rapporten.

