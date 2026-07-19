
WS.Namespace.Create('NovAxis.CheckBox', (function ()
{
    /**
    @class CheckBox
    @namespace NovAxis
    @constructor
    @param {JQuery dom} Element Dom element that will be replaced by the control. <br />
    Must be in the specific form : &lt;input type="checkbox"&gt;&lt;label&gt;Mon texte&lt;/label&gt;<br />
    Property checked intag is supported : &lt;input type="checkbox" <span style="background-color:yellow;">checked</span>&gt;&lt;label&gt;Mon texte&lt;/label&gt;
    @param {Boolean} [Value] true, false
    @param {Array<String>} [Classes] Array of class names to apply to the control.
    **/
    var Class = function (Element, Value, Classes)
    {
        // Validation
        if (!Element || Element.length == 0) throw "Element control cannot be null.";
        if (Value && (typeof Value !== "boolean")) throw "Value must be boolean.";

        /**
        Fired when the value is changed.

        @event change
        @param {Number} The new value.
        **/
        WS.EventObject.call(this, ['change']);

        // InitData
        this._Value = false;
        this.originalElement = Element;

        // Element
        this._Element = $(CreateControl.call(this, Element));

        Element.after(this._Element);
        Element.remove();
        if (Classes && Classes.length > 0) this._Element.addClass(Classes.join(" "));

        // Events, Bindings
        AddEvents.apply(this);
        if (Value != null) this.SetValue(Value);
    };

    WS.Exts.Inherits(Class, WS.EventObject);

    // Private Method(s)
    var CreateControl = function (Element)
    {
        var Html = [];
        var MyElement = Element.find("input[type=checkbox]");
        var Checked = "";
        if (MyElement.is("[checked]"))
        {
            Checked = " class=\"wsChecked\"";
            this._Value = true;
        }

        Html.push("<div class=\"naCheckBox\">");
        Html.push("<div" + Checked + "></div>");
        Html.push("<label>" + MyElement.next("label").text() + "</label>");
        Html.push("</div>");

        return Html.join("");
    };

    var AddEvents = function ()
    {
        var Me = this;
        var Element = this._Element.find("> div");
        var Labels = this._Element.find("> label");

        Element.off("click");
        Element.on("click", function ()
        {
            var Value = !Me._Value;
            Me.SetValue(Value);
            Me.RaiseEvent("change", Value);
        });

        Labels.off("click");
        Labels.on("click", function () { $(this).prev("div").click(); });
    };

    // Public Method(s)
    (function (Methods)
    {
        /**
        Get the selected value (checked).
    
        @method GetValue
        @return {boolean} Value
        **/
        Methods.GetValue = function ()
        {
            return this._Value;
        };

        /**
        Set the selected value (checked).
    
        @method SetValue
        @param {boolean} Value
        **/
        Methods.SetValue = function (Value)
        {
            if (Value == null) throw "Value cannot be null.";
            if (Value && (typeof Value !== "boolean")) throw "Value must be boolean.";

            this._Value = Value;
            var Element = this._Element.find("> div");
            Element.removeClass("wsChecked");
            if (this._Value) { Element.addClass("wsChecked"); }
        };

        /**
        Disable the control.
   
        @method Disable
        **/
        Methods.Disable = function ()
        {
            this._Element.find("> div").css({ "pointer-events": "none", "background-color": "#5c5c5c" });
            this._Element.find("> label").css({ "pointer-events": "none", "color": "#5c5c5c" });
        };

        /**
        Enable the control.
  
        @method Enable
        **/
        Methods.Enable = function ()
        {
            this._Element.find("> div").css({ "pointer-events": "", "background-color": "" });
            this._Element.find("> label").css({ "pointer-events": "", "color": "" });
        };

        /**
        Destroy the control.
 
        @method Destroy
        **/
        Methods.Destroy = function ()
        {
            this._Element.after(this.originalElement);
            this._Element.remove();
            this._Element = null;
        };

    })(Class.prototype);

    return Class;
})());