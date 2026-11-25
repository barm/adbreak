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

1. Not at all, and it's advertising: **commercial break**. A promo for another show, or a sponsor's billboard, is
   a commercial break when it fills the screen.
2. Shrunk to share the screen with an ad: **double box**.
3. Full screen, with advertising over it (graphic, banner or voice): **ad read**. A promo shown or read over the
   programme is an ad read, and so is a sponsor's logo or a "presented by" bug over it, even over a bumper shot
   of the venue or city.
4. Full screen, nothing advertised: **programme**. Content a sponsor paid for but that is programme itself (a
   "presented by" feature, the sponsor's name only in a segment title) is programme; the read or graphic naming the
   sponsor is an ad read.
5. The whole programme is an advertisement: **paid programming**.

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

Times are seconds from the start of the recording's media. Older files used `promo`, which reads as `ad`, and
`sponsor`, which reads as `double_box`.
