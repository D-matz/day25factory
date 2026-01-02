import gleam/dict.{type Dict}
import gleam/float
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
    1 -> 2
    2 -> 4
    3 -> 8
    4 -> 16
    5 -> 32
    6 -> 64
    7 -> 128
    8 -> 256
    9 -> 512
    10 -> 1024
    11 -> 2048
    //who knows if this actually speeds anything up but cant hurt
    _ -> 2 * pow_2(n - 1)
  }
}

//this is not how std lib works? idk maybe missed something
fn parse_num_to_float(s: String) {
  case s |> int.parse {
    Ok(i) -> Ok(i |> int.to_float)
    Error(_) -> s |> float.parse
  }
}

fn speed_float_to_sec(slider_value: Float) {
  case slider_value {
    -3.001 -> 0.0
    _ -> {
      let assert Ok(n) = int.power(10, slider_value)
      n
    }
  }
}

fn speed_float_to_step_millisec(slider_value: Float) {
  { speed_float_to_sec(slider_value) *. 1000.0 }
  |> float.round
}

// set of buttons (each a parity int) converted to parity int
// used for p1
fn parity(btns: Set(Button)) {
  set.fold(from: 0, over: btns, with: fn(acc, btn) {
    acc |> int.bitwise_exclusive_or(btn.bits)
  })
}

// combo of buttons, so can have >1 of a num, converted to parity int
// used for p2
fn levels_parity(l: Levels) {
  dict.fold(from: 0, over: l.num_ctrs, with: fn(acc, light, value) {
    case value % 2 == 0 {
      True -> acc
      False -> acc |> int.bitwise_exclusive_or(pow_2(light))
    }
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

pub type NumAmts {
  NumAmts(num_amts: Dict(Int, Int))
}

pub type HowManyButtonsInCombo {
  HowManyButtonsInCombo(buttons_used: Int)
}

pub type LevelButtonCombo {
  LevelButtonCombo(combo: Dict(NumAmts, HowManyButtonsInCombo))
}

pub type CombosForParity {
  CombosForParity(cfp: Dict(Int, LevelButtonCombo))
}

type LevelCost {
  CostUnknown
  CostImpossible
  CostVal(Int)
}

type Levels {
  Levels(num_ctrs: Dict(Int, Int))
}

// need to remember how much even parity combo cost to get to this level
// because of the awkward step by step
type LTCost {
  LTCost(child_tree: LevelTree, to_child: HowManyButtonsInCombo)
}

type LevelTree {
  LevelTree(
    levels: Levels,
    cost: LevelCost,
    children: List(LTCost),
    was_pulled_from_cache: Bool,
  )
}

type LevelsCostCache {
  LevelsCostCache(level_costs: Dict(Levels, LevelCost))
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
  OneLineLevel(
    combos: CombosForParity,
    tree: LevelTree,
    cost_for_levels: Option(LevelsCostCache),
  )
}

type M {
  OnOffM
  LevelM
}

type CostColors {
  CostColors(unknown: String, impossible: String, val: String)
}

type Model {
  Model(
    on_index: Int,
    all_done: Bool,
    colors: CostColors,
    onoff_or_level_input: M,
    cache_enabled: Bool,
    set_speed: Option(Float),
    default_speed_ms: Int,
    flip_tree: Bool,
    play: Bool,
    in: String,
    problems: List(String),
    count: Result(Int, String),
    out: Result(OneLine, String),
  )
}

fn init(starting_example) -> #(Model, Effect(Msg)) {
  let fake_starting_model_without_input_loaded =
    Model(
      on_index: 0,
      all_done: False,
      colors: CostColors(
        impossible: "#FF9999",
        unknown: "#FFFF99",
        val: "#99FF99",
      ),
      onoff_or_level_input: OnOffM,
      cache_enabled: False,
      set_speed: None,
      default_speed_ms: 0,
      flip_tree: False,
      play: True,
      count: Ok(0),
      problems: [],
      in: starting_example,
      out: Error("initial input not loaded"),
    )
  update_index_model_with_parsed_input(fake_starting_model_without_input_loaded)
}

// UPDATE ----------------------------------------------------------------------

type TimerTick {
  TimerTick(millisec: Int, index: Int)
}

type Msg {
  ClockTickedForward(TimerTick)
  UserEnteredInput(String)
  UserSelectedType(M)
  UserClickedReplay
  UserSwitchedCache
  UserSwitchedDefaultspeed
  UserSetSpeed(String)
  UserClickedStep
  UserFlippedTree
  UserSwitchedPlay
  UserSetUnknowncolor(String)
  UserSetImpossiblecolor(String)
  UserSetCalculatedcolor(String)
}

fn update(maybevalid: Model, msg: Msg) -> #(Model, Effect(Msg)) {
  case msg {
    ClockTickedForward(timer) -> {
      case maybevalid.out {
        Error(_) -> {
          #(maybevalid, effect.none())
        }
        Ok(m) -> {
          case timer.index == maybevalid.on_index {
            False -> #(maybevalid, effect.none())
            True -> {
              case m {
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
                      let new_ct = case maybevalid.count {
                        Error(_) -> maybevalid.count
                        Ok(ct) -> Ok(ct + set.size(check_combo))
                      }
                      add_count_go_next_line(maybevalid, timer.index, new_ct)
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
                            Model(
                              ..maybevalid,
                              out: Ok(curr_problem_next_combo),
                            ),
                            tick(timer, maybevalid.set_speed),
                          )
                        }
                        [] -> {
                          #(
                            Model(
                              ..maybevalid,
                              count: Error(
                                "no button combinations work for indicator lights goal "
                                <> "["
                                <> {
                                  goal_bits_parity
                                  |> int.to_base2
                                  |> string.to_graphemes()
                                  |> list.map(fn(bit) {
                                    case bit {
                                      "1" -> "#"
                                      "0" -> "."
                                      _ -> panic as "not zero??"
                                    }
                                  })
                                  |> string.concat
                                  |> fn(s) {
                                    case
                                      string.length(s) != list.length(num_bits)
                                    {
                                      True -> "." <> s
                                      False -> s
                                    }
                                  }
                                  |> string.reverse
                                }
                                <> "]",
                              ),
                            ),
                            effect.none(),
                          )
                        }
                      }
                    }
                  }
                }
                OneLineLevel(combos:, tree:, cost_for_levels:) -> {
                  case tree.cost {
                    CostUnknown -> {
                      let #(updated_tree, new_cache) =
                        lt_one_step(combos, tree, cost_for_levels)
                      let updated_line =
                        OneLineLevel(
                          combos:,
                          tree: updated_tree,
                          cost_for_levels: new_cache,
                        )
                      #(
                        Model(..maybevalid, out: Ok(updated_line)),
                        tick(timer, maybevalid.set_speed),
                      )
                    }
                    CostImpossible -> {
                      let new_ct = Error("impossible input: " <> maybevalid.in)
                      add_count_go_next_line(maybevalid, timer.index, new_ct)
                    }
                    CostVal(tree_min_ct) -> {
                      let new_ct = case maybevalid.count {
                        Error(_) -> maybevalid.count
                        Ok(ct) -> Ok(ct + tree_min_ct)
                      }
                      add_count_go_next_line(maybevalid, timer.index, new_ct)
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    UserSelectedType(newtype) -> {
      case newtype == maybevalid.onoff_or_level_input {
        True -> #(maybevalid, effect.none())
        False ->
          update_index_model_with_parsed_input(
            Model(
              ..maybevalid,
              on_index: maybevalid.on_index + 1,
              onoff_or_level_input: newtype,
            ),
          )
      }
    }
    UserClickedReplay ->
      update_index_model_with_parsed_input(
        Model(..maybevalid, on_index: maybevalid.on_index + 1),
      )
    UserSwitchedCache -> {
      update_index_model_with_parsed_input(
        Model(
          ..maybevalid,
          on_index: maybevalid.on_index + 1,
          cache_enabled: !maybevalid.cache_enabled,
        ),
      )
    }
    UserEnteredInput(input) ->
      update_index_model_with_parsed_input(
        Model(..maybevalid, in: input, on_index: maybevalid.on_index + 1),
      )
    // note changing speed does not restart the run
    // everything else does by calling update_index_model_with_parsed_input
    // setting speed does update index so that when you go from slow to fast it steps again immediately, but does not re parse input
    UserSwitchedDefaultspeed -> #(
      Model(..maybevalid, set_speed: case maybevalid.set_speed {
        Some(_) -> None
        None -> Some(0.0)
      }),
      tick(
        TimerTick(
          millisec: maybevalid.default_speed_ms,
          index: maybevalid.on_index + 1,
        ),
        None,
      ),
    )
    UserSetSpeed(spd) -> {
      case spd |> parse_num_to_float {
        Error(_) -> #(
          Model(
            ..maybevalid,
            out: Error(
              "strangely failed to parse step speed string from slider " <> spd,
            ),
          ),
          effect.none(),
        )
        Ok(s) -> #(
          Model(
            ..maybevalid,
            on_index: maybevalid.on_index + 1,
            set_speed: Some(s),
          ),
          //if user manually set some speed, won't use the model default ms
          tick(
            TimerTick(millisec: -123, index: maybevalid.on_index + 1),
            Some(s),
          ),
        )
      }
    }
    UserClickedStep -> todo as "idk 4"
    UserFlippedTree -> #(
      Model(..maybevalid, flip_tree: !maybevalid.flip_tree),
      effect.none(),
    )
    UserSwitchedPlay -> {
      let new_m =
        Model(
          ..maybevalid,
          play: !maybevalid.play,
          on_index: maybevalid.on_index + 1,
        )
      #(new_m, case new_m.play {
        True ->
          tick(
            TimerTick(millisec: new_m.default_speed_ms, index: new_m.on_index),
            new_m.set_speed,
          )
        False -> effect.none()
      })
    }
    UserSetUnknowncolor(rgb) -> #(
      Model(..maybevalid, colors: CostColors(..maybevalid.colors, unknown: rgb)),
      effect.none(),
    )
    UserSetImpossiblecolor(rgb) -> #(
      Model(
        ..maybevalid,
        colors: CostColors(..maybevalid.colors, impossible: rgb),
      ),
      effect.none(),
    )
    UserSetCalculatedcolor(rgb) -> #(
      Model(..maybevalid, colors: CostColors(..maybevalid.colors, val: rgb)),
      effect.none(),
    )
  }
}

// if making new child of zeros or less than zero
// set cost to 0 or impossible immediately
// we won't need to recurse later as it's not unknown
fn check_zeros_or_ltzero(lt: LevelTree) {
  let level_values =
    lt.levels.num_ctrs
    |> dict.to_list
    |> list.map(fn(tpl) { tpl.1 })
  let elt_ne_zero =
    level_values
    |> list.find(fn(n) { n != 0 })
  case elt_ne_zero {
    Error(_) -> LevelTree(..lt, cost: CostVal(0))
    Ok(_) -> {
      let elt_lt_zero =
        level_values
        |> list.find(fn(n) { n < 0 })
      case elt_lt_zero {
        Ok(_) -> LevelTree(..lt, cost: CostImpossible)
        Error(_) -> lt
      }
    }
  }
}

// if making new child with levels in cache
// set cost to cache level, which should be a value or impossible
fn check_cache(lt: LevelTree, optional_cache: Option(LevelsCostCache)) {
  case optional_cache {
    None -> lt
    Some(cache) ->
      case cache.level_costs |> dict.get(lt.levels) {
        Error(_) -> lt
        Ok(cached_cost) -> {
          LevelTree(..lt, cost: cached_cost, was_pulled_from_cache: True)
        }
      }
  }
}

fn levels_minus_combo_amt_then_halved(
  old_levels: Levels,
  check_btn_combo_child: NumAmts,
) -> Levels {
  Levels(
    old_levels.num_ctrs
    |> dict.map_values(fn(num, old_amt) {
      let subtract_combo = case
        check_btn_combo_child.num_amts |> dict.get(num)
      {
        Error(_) -> old_amt
        Ok(combo_amt) -> old_amt - combo_amt
      }
      subtract_combo / 2
    }),
  )
}

fn lt_one_step(
  precomputed_parity_combos: CombosForParity,
  lt: LevelTree,
  cache: Option(LevelsCostCache),
) -> #(LevelTree, Option(LevelsCostCache)) {
  case
    precomputed_parity_combos.cfp
    |> dict.get(lt.levels |> levels_parity)
  {
    Error(_) -> #(LevelTree(..lt, cost: CostImpossible), cache)
    Ok(pcs) -> {
      // child_not_solved might not be found
      // - in that case, we have all children and the cost val or impossible for each, so calc min for this one (1 step)
      // child_not_solved might be found
      // - in that case, button combo that takes us to a new child or child with cost unknown
      // -- if new child, just add it to children (1 step)
      // -- if existing child with cost unknown, recurse on the child until we get a new child or calculate new cost
      let child_not_solved =
        pcs.combo
        |> dict.to_list
        |> list.find(fn(tpl) {
          let combo_amts = tpl.0
          let new_levels =
            lt.levels |> levels_minus_combo_amt_then_halved(combo_amts)
          let check_child_missing_or_costunknown =
            lt.children
            |> list.find(fn(lt_child) {
              lt_child.child_tree.levels == new_levels
            })
          case check_child_missing_or_costunknown {
            Error(_) -> True
            Ok(found_child) -> found_child.child_tree.cost == CostUnknown
          }
        })
      case child_not_solved {
        Error(_) -> {
          //all children are solved -> can calc min for htis one
          let min_cost =
            list.fold(
              from: CostImpossible,
              over: lt.children,
              with: fn(acc, child) {
                case child.child_tree.cost {
                  CostImpossible -> acc
                  CostVal(child_cost) -> {
                    let our_cost_to_child =
                      child.to_child.buttons_used + { child_cost * 2 }
                    case acc {
                      CostUnknown -> panic as "we never set acc to cost unknown"
                      CostImpossible -> CostVal(our_cost_to_child)
                      CostVal(curr_cost) ->
                        CostVal(int.min(curr_cost, our_cost_to_child))
                    }
                  }
                  CostUnknown ->
                    panic as "should not be any children left with cost unknown"
                }
              },
            )
          let new_cache = case cache {
            None -> None
            Some(c) ->
              Some(LevelsCostCache(
                c.level_costs |> dict.insert(lt.levels, min_cost),
              ))
          }
          #(LevelTree(..lt, cost: min_cost), new_cache)
        }
        Ok(missing_lbc) -> {
          let look_for_levels =
            lt.levels |> levels_minus_combo_amt_then_halved(missing_lbc.0)
          let missing_or_costunknown =
            lt.children
            |> list.find(fn(lt_child) {
              lt_child.child_tree.levels == look_for_levels
            })
          let #(new_children, new_cache) = case missing_or_costunknown {
            Error(_) -> {
              // found a button combo that has correct parity but no child for the new level after subtracting those buttons
              // so create missing child with cost unknown
              let new_leveltree =
                LevelTree(
                  children: [],
                  cost: CostUnknown,
                  levels: look_for_levels,
                  was_pulled_from_cache: False,
                )
                |> check_zeros_or_ltzero
                //might set cost to 0 or impossible
                |> check_cache(cache)
              //if using cache, might set cost to some val
              let new_children = [
                LTCost(to_child: missing_lbc.1, child_tree: new_leveltree),
                ..lt.children
              ]
              #(new_children, cache)
            }
            Ok(ch) -> {
              // found a button combo for parity that does have a child with new levels, but cost unknown
              // recurse to add child
              let #(child_tree, new_cache) =
                lt_one_step(precomputed_parity_combos, ch.child_tree, cache)
              let existing_child_updated_cost =
                LTCost(to_child: ch.to_child, child_tree:)
              let new_children =
                lt.children
                |> list.map(fn(old_child) {
                  case old_child.child_tree.levels == look_for_levels {
                    True -> existing_child_updated_cost
                    False -> old_child
                  }
                })
              #(new_children, new_cache)
            }
          }
          #(LevelTree(..lt, children: new_children), new_cache)
        }
      }
    }
  }
}

fn tick(t: TimerTick, set_speed: Option(Float)) -> Effect(Msg) {
  use dispatch <- effect.from
  use <- set_timeout(case set_speed {
    Some(spd) -> spd |> speed_float_to_step_millisec
    None -> t.millisec
  })

  dispatch(ClockTickedForward(t))
}

@external(javascript, "./app.ffi.mjs", "set_timeout")
fn set_timeout(_delay: Int, _cb: fn() -> a) -> Nil {
  Nil
}

fn ticks_for_currline(l: OneLine) {
  case l {
    OneLineLevel(combos:, tree:, cost_for_levels:) -> {
      let sum_cfp =
        list.fold(
          from: 0,
          over: combos.cfp |> dict.values,
          with: fn(acc, num_combos) { acc + dict.size(num_combos.combo) },
        )
      int.min(1000, 300_000 / { sum_cfp * sum_cfp })
    }
    OneLineOnOff(
      btns:,
      goal_bits_parity:,
      num_bits:,
      combos:,
      check_combo:,
      check_combo_parity:,
    ) -> {
      let len = list.length(btns)
      case len < 8 {
        True -> 1000 / { len }
        False -> 1
      }
    }
  }
}

fn add_count_go_next_line(maybevalid: Model, timer_index: Int, add_count) {
  case maybevalid.problems {
    [] -> {
      #(
        case maybevalid.all_done {
          True -> maybevalid
          False -> Model(..maybevalid, all_done: True, count: add_count)
        },
        effect.none(),
      )
    }
    [next_line, ..rest] -> {
      let next =
        parse_line(
          next_line,
          maybevalid.cache_enabled,
          maybevalid.onoff_or_level_input,
        )
      case next {
        Ok(valid_next_problem) -> {
          let tick_delay_for_next_line = ticks_for_currline(valid_next_problem)
          #(
            Model(
              ..maybevalid,
              count: add_count,
              problems: rest,
              default_speed_ms: tick_delay_for_next_line,
              out: next,
            ),
            tick(
              TimerTick(index: timer_index, millisec: tick_delay_for_next_line),
              maybevalid.set_speed,
            ),
          )
        }
        Error(_) -> #(
          Model(..maybevalid, all_done: True, out: next, count: Error("")),
          effect.none(),
        )
      }
    }
  }
}

/// parses model.in and tries to set actual problem
/// also updates index, so new model's ticks will be received in update and old will be discarded
fn update_index_model_with_parsed_input(from_model: Model) {
  let input = from_model.in
  let problems =
    input
    |> string.split("\n")
    |> list.filter(fn(maybe_emptyline) { maybe_emptyline != "" })
  let from_model = Model(..from_model, problems:, all_done: False)
  add_count_go_next_line(from_model, from_model.on_index, Ok(0))
}

fn parse_line(
  line: String,
  cache_enabled: Bool,
  t: M,
) -> Result(OneLine, String) {
  let ret = case line |> string.split("]") {
    [] -> Error("error parsing " <> line <> " could not find \"]\"")
    [goal, ..rest] -> {
      case rest |> string.concat |> string.split("{") {
        [] -> Error("error parsing " <> line <> " could not find \"{\"")
        [buttons, ..goal_levels] -> {
          let user_btn_numlists =
            buttons
            |> string.trim
            |> string.replace("(", "")
            |> string.replace(")", "")
            |> string.split(" ")
            |> list.try_map(fn(nums) {
              nums
              |> string.split(",")
              |> list.try_map(fn(num) {
                case int.parse(num) {
                  Error(_) ->
                    Error(
                      "could not parse int from "
                      <> case num {
                        "" -> "empty string"
                        _ -> num
                      },
                    )
                  Ok(n) -> Ok(n)
                }
              })
            })
          case user_btn_numlists {
            Error(e) -> Error("parsing " <> line <> " got error " <> e)
            Ok(btn_numlists) ->
              Ok(case t {
                OnOffM -> {
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
                  case combos |> list.first() {
                    Error(_) ->
                      Error(
                        "no button combinations, which should not happen if at least 1 button is provided",
                      )
                    Ok(first_combo) -> {
                      let goal =
                        goal
                        |> string.to_graphemes
                        |> list.drop(1)
                        |> list.try_map(fn(c) {
                          case c {
                            "#" -> Ok(True)
                            "." -> Ok(False)
                            _ ->
                              Error(
                                "error parsing "
                                <> line
                                <> " indicator char "
                                <> c
                                <> " was not \".\" or \"#\"",
                              )
                          }
                        })
                      case goal {
                        Error(e) -> Error(e)
                        Ok(goal) ->
                          Ok({
                            let gb =
                              list.fold(
                                from: #(0, 1),
                                over: goal,
                                with: fn(acc, b) {
                                  let acc_now = case b {
                                    True -> acc.0 |> int.bitwise_or(acc.1)
                                    False -> acc.0
                                  }
                                  #(acc_now, acc.1 * 2)
                                },
                              )
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
                          })
                      }
                    }
                  }
                }
                LevelM -> {
                  let btn_parity_combos: CombosForParity =
                    list.fold(
                      from: CombosForParity(cfp: dict.new()),
                      over: all_combinations(btn_numlists),
                      with: fn(acc, combination) {
                        let num_amts =
                          list.fold(
                            from: dict.new(),
                            over: combination,
                            with: fn(outer_acc, btn) {
                              list.fold(
                                from: outer_acc,
                                over: btn,
                                with: fn(inner_acc, num) {
                                  let curr_val = inner_acc |> dict.get(num)
                                  case curr_val {
                                    Error(_) -> inner_acc |> dict.insert(num, 1)
                                    Ok(v) ->
                                      inner_acc |> dict.insert(num, v + 1)
                                  }
                                },
                              )
                            },
                          )
                        let new_cost =
                          HowManyButtonsInCombo(
                            buttons_used: combination |> list.length,
                          )
                        let new_amt = NumAmts(num_amts:)
                        let parity = Levels(num_amts) |> levels_parity
                        let existing = acc.cfp |> dict.get(parity)
                        let with_new_cfp = case existing {
                          Error(_) ->
                            dict.new() |> dict.insert(new_amt, new_cost)
                          Ok(others) -> {
                            case others.combo |> dict.get(new_amt) {
                              Error(_) ->
                                others.combo
                                |> dict.insert(new_amt, new_cost)
                              Ok(combo_same_total) ->
                                others.combo
                                |> dict.insert(
                                  new_amt,
                                  HowManyButtonsInCombo(int.min(
                                    combo_same_total.buttons_used,
                                    new_cost.buttons_used,
                                  )),
                                )
                            }
                          }
                        }
                        CombosForParity(
                          cfp: acc.cfp
                          |> dict.insert(parity, LevelButtonCombo(with_new_cfp)),
                        )
                      },
                    )
                  let goal_ints =
                    goal_levels
                    |> string.concat
                    |> string.drop_end(1)
                    |> string.split(",")
                    |> list.try_map(fn(num) {
                      case int.parse(num) {
                        Ok(n) -> Ok(n)
                        Error(_) ->
                          Error(
                            "parsing "
                            <> line
                            <> " could not parse int from "
                            <> case num {
                              "" -> "empty string"
                              _ -> num
                            },
                          )
                      }
                    })
                  case goal_ints {
                    Error(e) -> Error(e)
                    Ok(goal_ints) -> {
                      let starting_levels =
                        list.index_fold(
                          from: dict.new(),
                          over: goal_ints,
                          with: fn(acc, amt, idx) {
                            acc |> dict.insert(idx, amt)
                          },
                        )
                      let starting_tree =
                        LevelTree(
                          levels: Levels(starting_levels),
                          cost: CostUnknown,
                          children: [],
                          was_pulled_from_cache: False,
                        )
                      let starting_cache = case cache_enabled {
                        True -> Some(LevelsCostCache(level_costs: dict.new()))
                        False -> None
                      }
                      Ok(OneLineLevel(
                        combos: btn_parity_combos,
                        tree: starting_tree,
                        cost_for_levels: starting_cache,
                      ))
                    }
                  }
                }
              })
          }
        }
      }
    }
  }
  case ret {
    Ok(Ok(o)) -> Ok(o)
    Ok(Error(e)) -> Error(e)
    Error(e) -> Error(e)
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
      a.style("background", "#292d3e"),
      a.style("color", "#fefefc"),
      a.style("font-family", "verdana, helvetica, sans-serif"),
      a.style("height", "#100%"),
      a.style("margin", "0px"),
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
        h.text(" visualization: joltage indicators and levels "),
        h.a(
          [
            a.href(
              "https://github.com/D-matz/day25factory/blob/main/src/day25factory.gleam",
            ),
          ],
          [h.text("[source]")],
        ),
      ]),
      h.p([], case model.onoff_or_level_input {
        LevelM -> [
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
        OnOffM -> [
          h.text(
            "Pressing a button twice negates itself, i.e. does nothing. So a minimal solution presses each button 0 or 1 times only.",
          ),
        ]
      }),
      h.div(
        [
          a.style("display", "flex"),
          a.style("flex-direction", "column"),
        ],
        [
          h.div([], [
            h.input([
              a.type_("radio"),
              a.id("levels"),
              a.checked(case model.onoff_or_level_input {
                LevelM -> True
                OnOffM -> False
              }),
              event.on_click(UserSelectedType(LevelM)),
            ]),
            h.label([a.for("levels")], [h.text("joltage levels")]),
            h.span(
              [
                a.style("margin-left", "15px"),
                a.title(
                  "The first time we get minimum press count for some set of joltage levels, save that and use it next time we get to the same levels, instead of recursing. (un)checking this restarts everything to (not) build cached levels.",
                ),
              ],
              [
                h.input([
                  a.type_("checkbox"),
                  a.id("cache"),
                  a.disabled(case model.onoff_or_level_input {
                    LevelM -> False
                    OnOffM -> True
                  }),
                  a.checked(model.cache_enabled),
                  event.on_click(UserSwitchedCache),
                ]),
                h.label(
                  [
                    a.for("cache"),
                    a.style("opacity", case model.onoff_or_level_input {
                      LevelM -> "1.0"
                      OnOffM -> "0.3"
                    }),
                  ],
                  [h.text("cache levels")],
                ),
              ],
            ),
            h.span(
              [
                a.style("margin-left", "15px"),
                a.title(
                  "As joltage level tree lines are calculated, display them on top of screen instead of bottom.",
                ),
              ],
              [
                h.input([
                  a.type_("checkbox"),
                  a.id("flip"),
                  a.disabled(case model.onoff_or_level_input {
                    LevelM -> False
                    OnOffM -> True
                  }),
                  a.checked(model.flip_tree),
                  event.on_click(UserFlippedTree),
                ]),
                h.label(
                  [
                    a.for("flip"),
                    a.style("opacity", case model.onoff_or_level_input {
                      LevelM -> "1.0"
                      OnOffM -> "0.3"
                    }),
                  ],
                  [h.text("new level line at top")],
                ),
              ],
            ),
            h.span(
              [
                a.style("margin-left", "15px"),
                a.title(
                  "Set colors for joltage levels unknown, impossible, and calculated.",
                ),
              ],
              [
                h.input([
                  a.type_("color"),
                  a.value(model.colors.unknown),
                  a.id("u"),
                  a.style("margin-left", "3px"),
                  a.disabled(case model.onoff_or_level_input {
                    LevelM -> False
                    OnOffM -> True
                  }),
                  event.on_input(UserSetUnknowncolor),
                ]),
                h.label(
                  [
                    a.for("u"),
                    a.style("opacity", case model.onoff_or_level_input {
                      LevelM -> "1.0"
                      OnOffM -> "0.3"
                    }),
                  ],
                  [h.text("unknown")],
                ),
                h.input([
                  a.type_("color"),
                  a.value(model.colors.val),
                  a.id("c"),
                  a.style("margin-left", "3px"),
                  a.disabled(case model.onoff_or_level_input {
                    LevelM -> False
                    OnOffM -> True
                  }),
                  event.on_input(UserSetCalculatedcolor),
                ]),
                h.label(
                  [
                    a.for("c"),
                    a.style("opacity", case model.onoff_or_level_input {
                      LevelM -> "1.0"
                      OnOffM -> "0.3"
                    }),
                  ],
                  [h.text("calculated")],
                ),
                h.input([
                  a.type_("color"),
                  a.value(model.colors.impossible),
                  a.id("i"),
                  a.style("margin-left", "3px"),
                  a.disabled(case model.onoff_or_level_input {
                    LevelM -> False
                    OnOffM -> True
                  }),
                  event.on_input(UserSetImpossiblecolor),
                ]),
                h.label(
                  [
                    a.for("i"),
                    a.style("opacity", case model.onoff_or_level_input {
                      LevelM -> "1.0"
                      OnOffM -> "0.3"
                    }),
                  ],
                  [h.text("impossible")],
                ),
              ],
            ),
          ]),
          h.div([], [
            h.input([
              a.type_("radio"),
              a.id("lights"),
              a.checked(case model.onoff_or_level_input {
                LevelM -> False
                OnOffM -> True
              }),
              event.on_click(UserSelectedType(OnOffM)),
            ]),
            h.label([a.for("lights")], [h.text("indicator lights")]),
          ]),
        ],
      ),
      h.div(
        [
          a.style("display", "flex"),
          a.style("flex-wrap", "wrap"),
          a.style("gap", "15px"),
          a.style("align-items", "end"),
        ],
        [
          h.div(
            [
              a.style("display", "flex"),
              a.style("flex-direction", "column"),
            ],
            [
              h.input([
                a.type_("range"),
                a.min("-3.001"),
                a.max("3"),
                a.step("0.001"),
                a.disabled(case model.set_speed {
                  Some(_) -> False
                  None -> True
                }),
                a.value(case model.set_speed {
                  Some(n) -> n |> float.to_string
                  None -> "0"
                }),
                event.on_input(UserSetSpeed),
              ]),
              h.div([], [
                h.span(
                  [
                    a.title(
                      "The default step time is roughly a shorter duration for lines that look like they'll take more steps. However you can check this to manually set speed with slider.",
                    ),
                  ],
                  [
                    h.input([
                      a.type_("checkbox"),
                      a.id("defaultspeed"),
                      a.checked(case model.set_speed {
                        Some(_) -> True
                        None -> False
                      }),
                      event.on_click(UserSwitchedDefaultspeed),
                    ]),
                    h.label(
                      [
                        a.for("defaultspeed"),
                      ],
                      [h.text("manual step time ")],
                    ),
                    h.span(
                      [
                        a.style("width", "9ch"),
                        a.style("display", "inline-block"),
                      ],
                      [
                        h.text(case model.set_speed {
                          None -> "(off)"
                          Some(spd) ->
                            spd
                            |> speed_float_to_sec
                            |> float.to_precision(3)
                            |> float.to_string
                            <> "s"
                        }),
                      ],
                    ),
                  ],
                ),
              ]),
            ],
          ),
          h.button(
            [
              event.on_click(UserClickedReplay),
              a.style("width", "4em"),
              a.style("padding", "3px"),
            ],
            [
              h.text("replay"),
            ],
          ),
          h.button(
            [
              event.on_click(UserClickedStep),
              a.style("width", "4em"),
              a.style("padding", "3px"),
            ],
            [
              h.text("step"),
            ],
          ),
          h.button(
            [
              event.on_click(UserSwitchedPlay),
              a.style("width", "4em"),
              a.style("padding", "3px"),
            ],
            [
              case model.play {
                True -> h.text("pause")
                False -> h.text("play")
              },
            ],
          ),
          case model.out {
            Error(_) -> {
              h.text("")
            }
            Ok(_) -> {
              case model.count {
                Ok(n) ->
                  h.span([], [
                    h.text("min press sum: "),
                    h.span(
                      [
                        a.style("font-size", "1.4em"),
                        a.style("display", "inline-block"),
                      ],
                      [
                        h.text(n |> int.to_string),
                      ],
                    ),
                  ])
                Error(s) -> h.text(s)
              }
            }
          },
        ],
      ),
      h.textarea(
        [
          a.id("input"),
          a.style("height", "60px"),
          a.style("width", "480px"),
          event.on_input(UserEnteredInput),
        ],
        model.in,
      ),
      case model.out {
        Error(input_invalid) -> h.text(input_invalid)
        Ok(m) ->
          case m {
            OneLineOnOff(
              btns:,
              goal_bits_parity:,
              num_bits:,
              combos:,
              check_combo:,
              check_combo_parity:,
            ) -> {
              h.pre(
                [a.style("font-size", "3em"), a.style("overflow", "visible")],
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
                      ..list.fold(
                        from: [],
                        over: btns,
                        with: fn(outer_acc, btn) {
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
                        },
                      )
                    ]),
                  ),
              )
            }
            OneLineLevel(_, tree, _) ->
              h.pre(
                [],
                Root(tree) |> view_tree_lines(0, model.flip_tree, model.colors),
              )
          }
      },
    ],
  )
}

type TreeOrCost {
  Child(LTCost)
  Root(LevelTree)
}

fn view_tree_lines(
  node: TreeOrCost,
  indent: Int,
  flip_tree: Bool,
  colors: CostColors,
) -> List(Element(Msg)) {
  let #(lt, combo_cost) = case node {
    Child(ltc) -> #(
      ltc.child_tree,
      ltc.to_child.buttons_used |> int.to_string <> " + 2 * ",
    )
    Root(lt) -> #(lt, "")
  }
  let level_nums_string_before =
    h.text(string.concat([string.repeat("           ", indent), combo_cost]))
  let level_nums_string_color =
    h.span(
      [
        a.style("color", case lt.cost {
          CostImpossible -> colors.impossible
          CostUnknown -> colors.unknown
          CostVal(_) -> colors.val
        }),
      ],
      [
        h.text(
          string.concat([
            "{",
            lt.levels.num_ctrs
              |> dict.to_list
              |> list.sort(fn(d1, d2) { int.compare(d1.0, d2.0) })
              |> list.map(fn(tpl) { tpl.1 |> int.to_string })
              |> string.join(","),
            "}",
            " = ",
            case lt.cost {
              CostUnknown -> "?"
              CostImpossible -> "∞"
              CostVal(v) -> v |> int.to_string
            },
          ]),
        ),
      ],
    )
  let level_nums_string_after =
    h.text(
      string.concat([
        case lt.was_pulled_from_cache {
          True ->
            case lt.cost {
              CostUnknown -> " = min of"
              _ -> " (cached)"
            }
          False ->
            case lt.cost {
              CostUnknown -> " = min of"
              CostImpossible -> ""
              CostVal(v) ->
                case v {
                  0 -> ""
                  _ -> " = min of"
                }
            }
        },
        "\n",
      ]),
    )

  [
    level_nums_string_before,
    level_nums_string_color,
    level_nums_string_after,
    ..list.fold(from: [], over: lt.children, with: fn(acc, lt_child) {
      case flip_tree {
        False ->
          list.append(
            view_tree_lines(Child(lt_child), indent + 1, flip_tree, colors),
            acc,
          )
        True ->
          list.append(
            acc,
            view_tree_lines(Child(lt_child), indent + 1, flip_tree, colors),
          )
      }
    })
  ]
}
