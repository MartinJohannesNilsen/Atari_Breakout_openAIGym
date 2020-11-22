## DQN på OpenAI Gym `BreakoutDeterministic-v4`

En implementasjon av Deep Q-Network på retrospillet Atari Breakout.

----
### For å spille spillet selv

Vedlagt i kildekoden ligger det en fil `breakout_test.py` som gir deg muligheten til å spille spillet selv med tastaturet.
Dette er fra et av mine egne forsøk:

![24_meg](https://github.com/Martinnilsen99/Atari_Breakout_openAIGym/blob/master/ReadMe/Gifs/24_meg.gif)

Jeg klarte kun oppnå en score på 24, men før du bedømmer meg bør du prøve det ut selv, det var ikke så lett som det ser ut som!

---
### For å kjøre koden

Dette prosjektet tar i bruk pakken `argh`, som gjør det mulig å definere ulike parametere i terminalkommandoen for kjøring.
```Python
Grunnleggende:
$ python agent.py

Utvidelser:
-t, kjør i testmode med en rendret versjon med utgangspunkt i modell, uten trening
-l, kjør med lokal synkronisering mot wandb.ai
-n, definer navnet på kjøringen, hvis ikke vil dette bli spurt om
-c, sett en tidligere modell på formatet '.pth' som utgangspunkt 
```

---
### Forhåndsvisning av oppnådde poengsummer per kjøring:

Refererer her til [rapporten](https://martinnilsen.no/media/Maskinlæringsrapport_DQN_AtariBreakout.pdf) laget i forbindelse med prosjektet.

![381](https://github.com/Martinnilsen99/Atari_Breakout_openAIGym/blob/master/ReadMe/Gifs/381_increasedMaxLenFrom_368_30m_last_targ_model.gif)
