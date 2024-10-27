
/**
 * - ## `Calling`
 * ```TS
 * Called_Function(Function_Typifier, (anyProp: propType) =>  { 
 *      // Block of code here 
 * });
 * ```
 *  ### For example:
 * ```TS
 * window.addEventListener("keydown", (e: KeyboardEvent) => {
 *      if (e.ctrlkey && e.key === "s") alert ("Hotkey: CTRL + S")
 * })
 * ```
 */
export interface _Implement_Utils {
    Calling: {
        Called_Function?: Function,
        Function_Typifier: any
    }
}
