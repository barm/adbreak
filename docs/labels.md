# Labels

What the labels in `dataset/<id>/labels.json` mean, and how to place them. The labelling page (`adbreak label`)
writes them; `adbreak eval` and the model's training read them.

## Principle

A label says **what is on screen**, not what a viewer should do about it. Whether to skip a kind is the player's
decision, and may be a viewer's setting. The classifier's job is to say which kind each moment is.

## Kinds

Everything in reviewed time that isn't labelled is **programme**.

| Kind | `kind` | On screen | Examples |
|---|---|---|---|
| Programme | *(no label)* | The programme, with no advertising | Play, the studio, the show's scenes, credits |
| Commercial break | `ad` | The programme is gone; advertising fills the screen | Commercials, the network's promos for its own shows, station spots and bumpers, full-screen sponsor billboards ("brought to you by…" with no programme on screen) |
| Double box | `double_box` | The programme, shrunk beside or under an ad | A game in a corner box while an ad fills the rest (a squeeze-back) |
| Ad read | `ad_read` | The programme, full screen, with advertising over it or spoken over it | An announcer's "brought to you by…" after a break, a sponsor graphic or banner over play, "coming up next on…" over the programme |
| Paid programming | `paid_programming` | A programme that is itself an advertisement | An infomercial |

## Deciding

Ask, for each moment: **is the programme on screen, and how much of it?**

1. Not at all, and it's advertising: **commercial break**. A promo for another show is a commercial break when it
   fills the screen, and so is an ad that keeps only the programme's ticker or a sponsor bar below it.
2. Shrunk to share the screen with an ad: **double box**.
3. Full screen, with advertising over it (graphic, banner or voice): **ad read**. A promo shown or read over the
   programme is an ad read, and so is a sponsor's logo or a "presented by" bug over it, even over a bumper shot
   of the venue or city. A tease for the network's own show presented as a segment ("happening tonight") is an ad
   read too.
4. Full screen, nothing advertised: **programme**. That includes:
   - a small sponsor logo that stays on screen through a segment (in a news ticker, a score bug, a sponsored
     segment's title or set): an ad read is a moment of advertising, not a standing credit;
   - a segment in which the hosts present products to buy (a shopping segment);
   - content a sponsor paid for that is programme itself (a "presented by" feature). The read or graphic naming
     the sponsor is an ad read.
5. The whole programme is an advertisement: **paid programming**.

Sponsor billboards ("coverage presented by …", a sponsor's logo filling the screen) go with what they're next to:
part of the commercial break when one is beside them (even across the programme's own sting), an **ad read**
inside the programme otherwise, such as between a show's title and its first shot.

Two things that look like advertising but are programme:

- the programme teasing its own next segment ("coming up: …");
- small corner bugs during play (an app logo, a show's name, a replay's sponsor logo). A panel, banner or lower
  third that carries a message is an ad read.

## Placing edges

- An edge goes at the cut, to the picture. Where black or silence separates programme from advertising, put the
  edge in that gap.
- Kinds can touch: a break often ends straight into a double box or an ad read. Label each separately, edge to
  edge, rather than stretching one over the other.
- Mark what you've checked as **reviewed** (`R` on the labelling page). Only reviewed time is scored or trained
  on, so a recording can be labelled a part at a time, and unlabelled reviewed time really means programme.

## How each kind is used

| Kind | Break score (`adbreak eval`) | Model training |
|---|---|---|
| Commercial break | What a detector must find | Break |
| Double box | Left out; a break's edge anywhere across an adjacent double box counts as right. `eval` reports how much of it each source marked | Left out, for now |
| Ad read | Programme: marking it as break is a false positive. `eval` reports how much of it each source marked | Programme, for now |
| Paid programming | Left out | Left out |

Double box and ad read are planned as outputs of their own, once there are enough labelled.

## Format

```json
{
  "recording": { "...": "see adbreak_labels::Recording" },
  "reviewed": [{ "start": 0.0, "end": 5400.0 }],
  "breaks": [
    { "start": 812.4, "end": 1003.1, "kind": "ad" },
    { "start": 1003.1, "end": 1021.0, "kind": "ad_read" }
  ]
}
```

Times are seconds from the start of the recording's media.
