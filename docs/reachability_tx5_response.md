# Response to the Tx5 validation note (REACHABILITY_RUN (1).md)

Short version: **the Tx5 signature is correct.** Its low correlation with Tx1–Tx4 is
device_5's real carrier-frequency offset (CFO), not a wrong reference payload. The empirical
`eps_phys_empirical.npz` should stand as-is. Details and evidence below.

## The Tx5 signature is not over a different payload — it carries device_5's CFO

The note reads Tx5's 0.21 correlation with Tx1–Tx4 (which mutually correlate 0.95–0.99) as
proof that `T_eta5(c)` was built over a different reference payload `c`. It is not. Two facts:

1. **All five signatures use the identical preamble reference.** The 802.11 preamble/header is
   bit-identical across every device (verified: a received frame from any device cross-correlates
   equally, ~67–74, with *every* device's ideal reference). There is one common `c`; Tx5 uses the
   same one as the others.

2. **The 0.21 is a CFO phase ramp, and it lifts to 0.96 when the ramp is removed.** Fitting and
   removing a single linear phase from the Tx5 signature:
   - best-fit slope = **−0.00409 cyc/sample** — exactly device_5's measured CFO
     (independently found by the kappa-stage CFO search, which needed −0.004 to align device_5);
   - Tx5–Tx1 correlation: **0.21 → 0.96** after deramping.
   A different payload would not be repaired by removing a single linear phase; a CFO ramp is.
   The note's own "matched phase ~2 deg" is consistent with this: a *constant* phase match cannot
   fix a *ramp*, so the residual stays low — that is the signature of CFO, not payload mismatch.

device_5 is the victim and its CFO is the largest single-device offset in the set; that is why its
distorted signature decorrelates from the others over the ~480-sample preamble. The empirical
`eps_phys` retains it on purpose — CFO is part of the physical distance an adversary must cover.

## Why we keep the empirical construction, not parametric synthesis over common `c`

We did run the prescribed parametric synthesis (`T_eta_i(c)=Rapp_i(c+kappa_i·conj(c))` over one
common `c`, from each device's `(kappa_i, p_i, r_sat,i)`). It makes all five signatures correlate
1.00, but it is not sound as the physical bound:

- It **drops CFO**, the dominant fingerprint for this CVNN (established repeatedly in the project;
  device_5 and the D4/D5 pair are CFO-separated). A physical impersonation bound cannot omit the
  impairment the classifier actually keys on.
- Its rho only *nearly* passes (rho ≤ 1 for all pairs **except Tx3→Tx2 = 1.12**), and that near-pass
  is an **artifact of the per-device Rapp fits**, which are fit-noise: Rapp is not
  device-discriminative here (received PAPR is 0.46–0.52 dB below ideal, uniform across devices,
  spread 0.06 dB; the `(p,r_sat)` fit is unstable with residual ≈ the effect). device_3's
  `r_sat=3.66` vs the others' `~7.3` is noise; it is exactly that spurious gap that inflates the
  Tx3 eps_phys enough to pull most pairs under 1. Kappa-only synthesis returns rho to 6.7.

## What actually holds

With the empirical CFO-complete `eps_phys_empirical.npz`, **rho ≤ 1 for all 20 pairs, max 0.172**
(Tx3→Tx2). eps_phys is 0.65–3.2; device_5 (Tx5) ~3.15 from its CFO; devices 2–5 ~0.65–0.93, which
sit at the within-device split-half floor (~0.90), so for the similar devices this is a
noise-floor-limited lower estimate rather than a crisp distance — flagged, not hidden.

Underlying open point (independent of Tx5): **device_3 (Tx3)** is separated by the CVNN with the
largest radii in the matrix, yet its kappa *and* CFO are close to the others. No simple parametric
impairment (kappa, Rapp, CFO) explains device_3's separation, so any parametric eps_phys
underestimates it and yields rho > 1 there. The empirical distance keeps rho ≤ 1, but device_3's
separation mechanism is worth a dedicated look before the security paragraph leans hard on it.

## Bottom line

- Tx5 needs no rerun; the signature is correct (CFO, verified).
- `eps_phys_empirical.npz` stands; rho ≤ 1 everywhere, max 0.172.
- kappa and R are validated and unchanged.
- Do not adopt kappa+Rapp parametric synthesis as the bound — it omits CFO and its pass depends on
  non-identifiable Rapp noise.
