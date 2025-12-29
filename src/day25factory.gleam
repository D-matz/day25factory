import gleam/dict.{type Dict}
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/set.{type Set}
import gleam/string
import lustre
import lustre/attribute as a
import lustre/effect.{type Effect}
import lustre/element.{type Element}
import lustre/element/html as h
import lustre/event

fn pow_2(n: Int) {
  case n {
    0 -> 1
    _ -> 2 * pow_2(n - 1)
  }
}

//set of buttons (each a parity int) converted to parity int
fn parity(btns: Set(Button)) {
  set.fold(from: 0, over: btns, with: fn(acc, btn) {
    acc |> int.bitwise_exclusive_or(btn.bits)
  })
}

//list of buttons (each a list of nums) converted to parity int
fn parity_button_list(btns: List(List(Int))) {
  list.fold(from: 0, over: btns, with: fn(outer_acc, btn) {
    list.fold(from: outer_acc, over: btn, with: fn(inner_acc, num) {
      inner_acc |> int.bitwise_exclusive_or(pow_2(num))
    })
  })
}

pub fn main() {
  let app = lustre.application(init, update, view)
  let assert Ok(_) = lustre.start(app, "#app", init_input)

  Nil
}

const init_input = "[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}
[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}"

// MODEL -----------------------------------------------------------------------

type Button {
  Button(bits: Int)
}

type LevelCost {
  CostUnknown
  CostImpossible
  CostVal(Int)
}

type LevelTree {
  LevelTree(levels: List(Int), cost: LevelCost, leaves: List(LevelTree))
}

type LevelButton {
  LevelButton(nums: List(Int))
}

type LevelButtonCombo {
  LevelButtonCombo(combo: List(LevelButton))
}

type LevelButtonParityCombos {
  LevelButtonParityCombos(parity_combos: List(LevelButtonCombo))
}

type OneLine {
  OneLineOnOff(
    btns: List(Button),
    goal_bits_parity: Int,
    num_bits: List(Int),
    combos: List(Set(Button)),
    check_combo: Set(Button),
    check_combo_parity: Int,
  )
  OneLineLevel(combos: Dict(Int, LevelButtonParityCombos), explored: LevelTree)
}

type Model {
  Model(
    in: String,
    curr_problem: OneLine,
    problems: List(OneLine),
    on_index: Int,
    count: Result(Int, String),
  )
}

fn init(starting_example) -> #(Model, Effect(Msg)) {
  next_model(starting_example, 0, OnOffs)
}

// UPDATE ----------------------------------------------------------------------

type TimerTick {
  TimerTick(millisec: Int, index: Int)
}

type Msg {
  ClockTickedForward(TimerTick)
  UpdateSolvedLine(TimerTick)
  UserEnteredInput(String)
  UserClickedPower
  UserClickedReplay
}

fn update(m: Model, msg: Msg) -> #(Model, Effect(Msg)) {
  case msg {
    ClockTickedForward(timer) -> {
      case timer.index == m.on_index {
        False -> #(m, effect.none())
        True -> {
          case m.curr_problem {
            OneLineOnOff(
              btns:,
              goal_bits_parity:,
              num_bits:,
              combos:,
              check_combo:,
              check_combo_parity:,
            ) -> {
              case check_combo_parity == goal_bits_parity {
                True -> {
                  //this case to check if end sort of awkwardly both here +
                  case m.problems {
                    [] -> {
                      #(
                        Model(..m, count: case m.count {
                          Error(_) -> m.count
                          Ok(ct) -> Ok(ct + set.size(check_combo))
                        }),
                        effect.none(),
                      )
                    }
                    [_, ..] -> {
                      #(
                        Model(..m, count: case m.count {
                          Error(_) -> m.count
                          Ok(ct) -> Ok(ct + set.size(check_combo))
                        }),
                        tick_plus_delay(timer, 1000),
                      )
                    }
                  }
                }
                False -> {
                  case combos {
                    [fst_combo, ..rest_combos] -> {
                      let curr_problem_next_combo =
                        OneLineOnOff(
                          check_combo: fst_combo,
                          check_combo_parity: fst_combo |> parity,
                          combos: rest_combos,
                          btns:,
                          goal_bits_parity:,
                          num_bits:,
                        )
                      #(
                        Model(..m, curr_problem: curr_problem_next_combo),
                        tick(timer),
                      )
                    }
                    [] -> {
                      todo
                    }
                  }
                }
              }
            }
            OneLineLevel(combos:, explored:) -> {
              todo
            }
          }
        }
      }
    }
    UserClickedPower -> {
      next_model(m.in, m.on_index + 1, case m.curr_problem {
        OneLineLevel(_, _) -> OnOffs
        OneLineOnOff(_, _, _, _, _, _) -> Levels
      })
    }
    UserClickedReplay -> {
      next_model(m.in, m.on_index + 1, case m.curr_problem {
        OneLineOnOff(_, _, _, _, _, _) -> OnOffs
        OneLineLevel(_, _) -> Levels
      })
    }
    UserEnteredInput(input) -> {
      next_model(input, m.on_index + 1, case m.curr_problem {
        OneLineOnOff(_, _, _, _, _, _) -> OnOffs
        OneLineLevel(_, _) -> Levels
      })
    }
    //after solving line wait a sec before going to next line to show solution
    UpdateSolvedLine(timer) -> {
      case m.curr_problem {
        OneLineOnOff(
          btns:,
          goal_bits_parity:,
          num_bits:,
          combos:,
          check_combo:,
          check_combo_parity:,
        ) -> {
          // case m.problems {
          //   [] -> {
          //     #(
          //       Model(..m, count: case m.count {
          //         Error(_) -> m.count
          //         Ok(ct) -> Ok(ct + set.size(check_combo))
          //       }),
          //       effect.none(),
          //     )
          //   }

          // should only go to UpdateSolvedLine if there are more lines left in problems
          // awkward split of this + case where tick_plus_delay is called
          // because on finishing problem we want to update counter only, then pause 1s, then update to next problem
          let assert [next, ..rest] = m.problems
          #(
            Model(
              in: m.in,
              curr_problem: next,
              problems: rest,
              on_index: m.on_index,
              count: m.count,
            ),
            tick(TimerTick(
              index: timer.index,
              millisec: ticks_for_len(num_bits),
            )),
          )
        }
        OneLineLevel(combos:, explored:) -> todo
      }
    }
  }
}

fn tick_plus_delay(t: TimerTick, delay: Int) -> Effect(Msg) {
  use dispatch <- effect.from
  use <- set_timeout(t.millisec + delay)

  dispatch(UpdateSolvedLine(t))
}

fn tick(t: TimerTick) -> Effect(Msg) {
  use dispatch <- effect.from
  use <- set_timeout(t.millisec)

  dispatch(ClockTickedForward(t))
}

@external(javascript, "./app.ffi.mjs", "set_timeout")
fn set_timeout(_delay: Int, _cb: fn() -> a) -> Nil {
  Nil
}

fn next_model(input: String, index: Int, t: M) {
  let m = new_model(input, index, t)
  case m.curr_problem {
    OneLineLevel(combos:, explored:) -> todo
    OneLineOnOff(
      btns:,
      goal_bits_parity:,
      num_bits:,
      combos:,
      check_combo:,
      check_combo_parity:,
    ) -> #(m, tick(TimerTick(ticks_for_len(num_bits), index)))
  }
}

fn ticks_for_len(num_bits) {
  let len = list.length(num_bits)
  case len < 8 {
    True -> 3000 / { len |> pow_2 }
    False -> 1
  }
}

type M {
  OnOffs
  Levels
}

fn new_model(input: String, curr_index: Int, t: M) {
  let problems =
    input
    |> string.split("\n")
    |> list.map(fn(line) {
      let assert [goal, ..rest] = line |> string.split("]")
      let assert [buttons, ..] = rest |> string.concat |> string.split("{")
      let btn_numlists =
        buttons
        |> string.replace("(", "")
        |> string.replace(")", "")
        |> string.trim()
        |> string.split(" ")
        |> list.map(fn(nums) {
          nums
          |> string.split(",")
          |> list.map(fn(num) {
            let assert Ok(n) = int.parse(num)
            n
          })
        })
      case t {
        OnOffs -> {
          let btns =
            btn_numlists
            |> list.map(fn(btn) {
              list.fold(
                from: Button(bits: 0),
                over: btn,
                with: fn(btn_acc, num) {
                  Button(btn_acc.bits |> int.bitwise_or(pow_2(num)))
                },
              )
            })
          let combos = all_combinations(btns) |> list.map(set.from_list)
          let assert Ok(first_combo) = combos |> list.first()
          let goal =
            goal
            |> string.to_graphemes
            |> list.drop(1)
            |> list.map(fn(c) {
              case c {
                "#" -> True
                "." -> False
                _ -> panic
              }
            })
          let gb =
            list.fold(from: #(0, 1), over: goal, with: fn(acc, b) {
              let acc_now = case b {
                True -> acc.0 |> int.bitwise_or(acc.1)
                False -> acc.0
              }
              #(acc_now, acc.1 * 2)
            })
          let goal_bits_parity = gb.0
          let num_bits = list.range(0, list.length(goal) - 1)
          OneLineOnOff(
            goal_bits_parity:,
            btns:,
            num_bits:,
            combos:,
            check_combo: first_combo,
            check_combo_parity: first_combo |> parity,
          )
        }
        Levels -> {
          let btn_parity_combos =
            list.fold(
              from: dict.new(),
              over: all_combinations(btn_numlists),
              with: fn(acc, combo) {
                let parity = combo |> parity_button_list
                let existing = acc |> dict.get(parity)
                let new = case existing {
                  Error(_) -> [combo]
                  Ok(others) -> [combo, ..others]
                }
                acc |> dict.insert(parity, new)
              },
            )

          let starting_level =
            rest
            |> string.concat
            |> string.drop_end(1)
            |> string.split(",")
            |> list.map(fn(x) {
              let assert Ok(n) = int.parse(x)
              n
            })
          OneLineLevel(combos: todo, explored: todo)
        }
      }
    })
  case problems {
    [first_problem, ..rest_problems] ->
      Model(
        in: input,
        problems: rest_problems,
        curr_problem: first_problem,
        on_index: curr_index,
        count: Ok(0),
      )
    [] ->
      Model(
        in: input,
        problems: [],
        curr_problem: OneLineOnOff(
          btns: [],
          goal_bits_parity: 0,
          num_bits: [],
          combos: [],
          check_combo: set.new(),
          check_combo_parity: 0,
        ),
        on_index: curr_index,
        count: Error("empty input (need at least 1 line)"),
      )
  }
}

fn all_combinations(l) {
  list.fold(from: [], over: list.range(0, l |> list.length), with: fn(acc, n) {
    acc
    |> list.append(l |> list.combinations(n))
  })
}

// VIEW ------------------------------------------------------------------------

fn view(model: Model) -> Element(Msg) {
  h.article(
    [
      a.style("display", "flex"),
      a.style("flex-direction", "column"),
      a.style("gap", "5px"),
    ],
    [
      h.h1([], [h.a([a.href("/")], [h.text("Correct Arity")])]),
      h.p([], [
        h.a([a.href("https://adventofcode.com/2025/day/10")], [
          h.text("Advent of Code day 10"),
        ]),
        h.text(" visualization: joltage indicator and level "),
        h.a(
          [
            a.href(
              "https://github.com/D-matz/day25tachyon/blob/main/src/tachyon.gleam",
            ),
          ],
          [h.text("(source todo)")],
        ),
      ]),
      h.p([], case False {
        True -> [
          h.text("From "),
          h.a(
            [
              a.href(
                "https://old.reddit.com/r/adventofcode/comments/1pk87hl/2025_day_10_part_2_bifurcate_your_way_to_victory/",
              ),
            ],
            [h.text("u/tenthmascot")],
          ),
          h.text(
            ": for all combos of 0 or 1 of each button making all counters even, divide by 2 and recurse.",
          ),
        ]
        False -> [
          h.text(
            "Pressing a button twice negates itself and does nothing, so a minimal solution presses each button at most once.",
          ),
        ]
      }),
      h.div([], [
        h.button(
          [
            event.on_click(UserClickedPower),
            a.style("width", "115px"),
            a.style("padding", "3px"),
          ],
          [
            h.text(case model.curr_problem {
              OneLineLevel(_, _) -> "set indicators"
              OneLineOnOff(_, _, _, _, _, _) -> "set joltage levels"
            }),
          ],
        ),
        h.button(
          [
            event.on_click(UserClickedReplay),
            a.style("width", "fit-content"),
            a.style("margin", "5px"),
            a.style("padding", "3px"),
          ],
          [
            h.text("Replay"),
          ],
        ),
        h.text(
          // case model.curr_b {
          //   Normal(_) -> "split count: "
          //   Quantum(_) -> "timeline count: "
          // }
          "min press sum: "
          <> case model.count {
            Ok(n) -> n |> int.to_string
            Error(s) -> s
          },
        ),
      ]),
      h.textarea(
        [
          a.id("input"),
          a.style("height", "60px"),
          a.style("width", "480px"),
          event.on_input(UserEnteredInput),
        ],
        model.in,
      ),
      case model.curr_problem {
        OneLineOnOff(
          btns:,
          goal_bits_parity:,
          num_bits:,
          combos:,
          check_combo:,
          check_combo_parity:,
        ) -> {
          h.pre(
            [a.style("font-size", "3em")],
            list.reverse(
              list.fold(from: [], over: num_bits, with: fn(light_acc, bit) {
                let light_on =
                  check_combo_parity
                  |> int.bitwise_and(pow_2(bit))
                case light_on {
                  0 -> [
                    h.span(
                      [
                        a.style("width", "3ch"),
                        a.style("display", "inline-block"),
                        a.style("text-align", "center"),
                        a.style("opacity", "0.35"),
                      ],
                      [
                        h.text("💡"),
                      ],
                    ),
                    ..light_acc
                  ]
                  _ -> [
                    h.span(
                      [
                        a.style("width", "3ch"),
                        a.style("display", "inline-block"),
                        a.style("text-align", "center"),
                        a.style("filter", "drop-shadow(0 0 25px yellow)"),
                      ],
                      [
                        h.text("💡"),
                      ],
                    ),
                    ..light_acc
                  ]
                }
              }),
            )
              |> list.append(
                list.reverse([
                  h.text(
                    num_bits
                    |> list.map(fn(n) {
                      case n < 10 {
                        True -> " " <> int.to_string(n) <> " "
                        False ->
                          case n < 100 {
                            True -> int.to_string(n) <> " "
                            False ->
                              case n < 1000 {
                                True -> int.to_string(n)
                                False -> int.to_string(n % 1000)
                              }
                          }
                      }
                    })
                    |> string.concat,
                  ),
                  h.text("\n"),
                  ..list.fold(from: [], over: btns, with: fn(outer_acc, btn) {
                    let btn_on = check_combo |> set.contains(btn)
                    [
                      h.text(
                        string.concat([
                          "(",
                          num_bits
                            |> list.filter(fn(n) {
                              case pow_2(n) |> int.bitwise_and(btn.bits) {
                                0 -> False
                                _ -> True
                              }
                            })
                            |> list.map(int.to_string)
                            |> string.join(","),
                          ")",
                        ]),
                      ),
                      ..list.fold(
                        from: [h.text("\n"), ..outer_acc],
                        over: num_bits,
                        with: fn(inner_acc, bit) {
                          let new_elt = case
                            btn.bits |> int.bitwise_and(pow_2(bit))
                          {
                            0 -> h.text("   ")
                            _ ->
                              case btn_on {
                                True -> h.text("[x]")
                                False -> h.text("[ ]")
                              }
                          }
                          [new_elt, ..inner_acc]
                        },
                      )
                    ]
                  })
                ]),
              ),
          )
        }
        OneLineLevel(combos:, explored:) -> h.pre([], [h.text("power levels")])
      },
    ],
  )
}
