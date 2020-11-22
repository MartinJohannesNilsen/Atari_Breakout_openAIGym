# DQN på Atari Breakout

En implementasjon av Deep Q-Network på retrospillet Atari Breakout.  
Det ble skrevet en rapport i forbindelse med prosjektet, som er tilgjengelig [her](https://martinnilsen.no/media/Maskinlæringsrapport_DQN_AtariBreakout.pdf).  
Miljøet `BreakoutDeterministic-v4` fra OpenAI Gym ble tatt i bruk, implementert med bla. Python og PyTorch.

----
## Spill spillet selv

Vedlagt i kildekoden ligger det en fil `breakout_test.py` som gir deg muligheten til å spille spillet selv med tastaturet.  
Dette er fra et av mine egne forsøk:

![24_meg](https://github.com/Martinnilsen99/Atari_Breakout_openAIGym/blob/master/ReadMe/Gifs/24_meg.gif)

Jeg klarte kun oppnå en score på 24, men før du bedømmer meg bør du prøve det ut selv, det var ikke så lett som det ser ut som!

---
## Kjøre koden

Dette prosjektet tar i bruk pakken `argh`, som gjør det mulig å definere ulike parametere i terminalkommandoen for kjøring.
```Python
Grunnleggende:
$ python agent.py

Utvidelser:
-t, kjør i testmodus der du ser en rendret versjon med utgangspunkt i modell, uten trening
-l, kjør med lokal synkronisering mot wandb.ai
-n <"Navn">, definer navnet på kjøringen (vil bli spurt om dersom ikke testmodus)
-c <"model.pth">, sett en tidligere modell i **Models** på formatet '.pth' som utgangspunkt
```

En kan for eksempel kjøre `$ python agent.py -c "rmsprop_før_379.pth"` for å kjøre en modell som klarer få en poengsum på 360.  
Ettersom systemet er deterministisk skal du få det samme ved kjøring.

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

Boltzmann med redusert handlingsrom. Score på 368 etter 30m steg, og mister ikke siste livet. Ble avsluttet etter den nådde maxgrensa for antall handlinger i en epoke, en begrensning satt i det originale miljøet for å unngå at den kjører uendelig dersom agenten bla. ikke starter spillet selv.

## En gang det ikke gikk like bra

![0_22m_boltzeps.gif](https://github.com/Martinnilsen99/Atari_Breakout_openAIGym/blob/master/ReadMe/Gifs/0_22m_boltzeps.gif "Boltzmann med synkende epsilon. Uendelig løkke ettersom den ikke starter spillet.")

Et eksempel på nevnt scenario over. Agenten starter ikke spillet, men vibrerer bare ved å gå hurtig fra høyre til venstre.

## Til sammenlikning: Første kjøring

![36_15m_v0.gif](https://github.com/Martinnilsen99/Atari_Breakout_openAIGym/blob/master/ReadMe/Gifs/36_15m_v0.gif "Første fungerende kjøring i Breakout-v0. Score på 36 etter 15m steg.")

Første fungerende kjøring i miljløet *Breakout-v0*. Oppnådd score på 36 etter 15m steg. Dette var siste kjøring der dette miljøet ble tatt i bruk, videre ble miljøet *BreakoutDeterministic-v4* brukt. Begrunnelse for dette valget står beskrevet i rapporten.

